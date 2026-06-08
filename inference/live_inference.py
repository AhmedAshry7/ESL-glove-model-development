import asyncio
import websockets
import struct
import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import threading
import time
import msvcrt

try:
    import arabic_reshaper
    from bidi.algorithm import get_display
    def fix_arabic(text):
        return get_display(arabic_reshaper.reshape(text))
except ImportError:
    def fix_arabic(text):
        return text
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.table import Table
from rich.console import Group

# Add project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_dir)

from inference.sliding_window_recognizer import SlidingWindowRecognizer
import config.pipeline_config as cfg
from preprocessing.stream_preprocessor import preprocess_stream

# --- Relay server (port 8766) — broadcasts predictions to connected mobile clients ---
RELAY_PORT = 8766
_relay_clients: set = set()
_bg_tasks: set = set()

def _task_done(task, ui_ref=None):
    _bg_tasks.discard(task)
    exc = task.exception()
    if exc and ui_ref:
        ui_ref.log_relay(f"[CRITICAL] Broadcast crashed: {type(exc).__name__} - {exc}")

async def _relay_handler(websocket, path=None, ui_ref=None):
    """Handle a single relay client; keep it alive until it disconnects."""
    _relay_clients.add(websocket)
    if ui_ref: ui_ref.log_relay(f"[+] Phone connected. Total: {len(_relay_clients)}")
    try:
        async for _ in websocket:
            pass
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        _relay_clients.discard(websocket)
        if ui_ref: ui_ref.log_relay(f"[-] Phone disconnected. Total: {len(_relay_clients)}")

async def _broadcast(word: str, ui_ref=None):
    """Broadcast a detected word as JSON to all connected relay clients."""
    global _relay_clients
    if not _relay_clients:
        if ui_ref: ui_ref.log_relay("Relay: No phones connected.")
        return
    payload = json.dumps({"word": word, "sentence": word})
    dead = set()
    for ws in set(_relay_clients):
        try:
            await ws.send(payload)
        except Exception:
            dead.add(ws)
    _relay_clients -= dead
    if ui_ref:
        ui_ref.log_relay(f"[>] Sent: '{word}' ({len(_relay_clients)} phones)")

# Hardcoded REST_POSES extracted from HumanCharacterDummy_M.glb
REST_POSES = {
  "B-upperArm.R": {
    "local": [0.0237012058496475, -0.70952320098877, 0.0732292383909225, 0.700465977191925],
    "world": [0.009204285533778778, 0.003129380105630894, 0.7316245332703986, 0.6816385222251816]
  },
  "B-forearm.R": {
    "local": [0.00103800499346107, 0.000423331861384213, -0.00320175429806113, 0.999994218349457],
    "world": [0.009592037025914301, 0.004206821096695134, 0.7294385123599116, 0.6839661843627645]
  },
  "B-hand.R": {
    "local": [0.0136553505435586, -0.705368518829346, -0.0330981500446796, 0.707935929298401],
    "world": [0.530514070697742, -0.4691918373238772, 0.4869343491517566, 0.5111836782039316]
  },
  "B-upperArm.L": {
    "local": [0.0238260049372911, 0.709510564804077, -0.0733541920781136, 0.700461447238922],
    "world": [0.009325400903213044, -0.0032585972235789715, -0.731623276986052, 0.6816375970817109]
  },
  "B-forearm.L": {
    "local": [0.000659474404528737, -0.00042493743239902, 0.00320027419365942, 0.999994575977325],
    "world": [0.00945355034905899, -0.004060563543990103, -0.7294396951838917, 0.6839677603936022]
  },
  "B-hand.L": {
    "local": [0.0137976342812181, 0.705362558364868, 0.0332405641674995, 0.707932472229004],
    "world": [0.5305140864327645, 0.46919186103064126, -0.48693436589504024, 0.5111837073938759]
  }
}

def to_R(quat_array):
    if np.any(np.isnan(quat_array)):
        return R.identity()
    return R.from_quat(quat_array) # [x, y, z, w]

def ConvertToThreeSpace(q_raw, hand='right'):
    x, y, z, w = q_raw
    return np.array([-z, -x, y, w])

class GloveCalibrator:
    def __init__(self):
        self.is_calibrated = False
        
        self.upperRestR = to_R(REST_POSES["B-upperArm.R"]["local"])
        self.upperRestWorldR = to_R(REST_POSES["B-upperArm.R"]["world"])
        self.forearmRestR = to_R(REST_POSES["B-forearm.R"]["local"])
        self.handRestR = to_R(REST_POSES["B-hand.R"]["local"])

        self.upperRestWorldL = to_R(REST_POSES["B-upperArm.L"]["world"])
        self.forearmRestL = to_R(REST_POSES["B-forearm.L"]["local"])
        self.handRestL = to_R(REST_POSES["B-hand.L"]["local"])
        
        # Initialize identity mounts for pre-calibration output
        identity = to_R([0.0, 0.0, 0.0, 1.0])
        self.tareR = identity
        self.upperMountCorrR = identity
        self.forearmMountLR = identity
        self.forearmMountRR = identity
        self.handMountLR = identity
        self.handMountRR = identity
        
        self.tareL = identity
        self.upperMountCorrL = identity
        self.forearmMountLL = identity
        self.forearmMountRL = identity
        self.handMountLL = identity
        self.handMountRL = identity

    def calibrate(self, r_imu, l_imu):
        # Right Arm
        hwUpR = to_R(ConvertToThreeSpace(r_imu['upperArm'], 'right'))
        hwFoR = to_R(ConvertToThreeSpace(r_imu['forearm'], 'right'))
        hwHaR = to_R(ConvertToThreeSpace(r_imu['hand'], 'right'))

        # deltaR = hwUpR * self.upperRestR.inv()
        # dqR = deltaR.as_quat()
        # tareQ_arrR = np.array([0, dqR[1], 0, dqR[3]])
        # if np.linalg.norm(tareQ_arrR) < 0.0001:
        #     tareQ_arrR = np.array([0, 1, 0, 0])
        # tareQ_arrR = tareQ_arrR / np.linalg.norm(tareQ_arrR)
        # self.tareR = to_R(tareQ_arrR)
        eulerR = hwUpR.as_euler('YXZ',degrees = False)
        self.tareR = R.from_euler('XYZ',[0,eulerR[0],0],degrees = False)

        self.upperMountCorrR = hwUpR.inv() * self.tareR * self.upperRestWorldR
        self.forearmMountLR = self.upperMountCorrR.inv()
        self.forearmMountRR = hwFoR.inv() * self.upperMountCorrR * self.forearmRestR
        self.handMountLR = self.forearmMountRR.inv()
        self.handMountRR = hwHaR.inv() * self.forearmMountRR * self.handRestR

        # Left Arm
        hwUpL = to_R(ConvertToThreeSpace(l_imu['upperArm'], 'left'))
        hwFoL = to_R(ConvertToThreeSpace(l_imu['forearm'], 'left'))
        hwHaL = to_R(ConvertToThreeSpace(l_imu['hand'], 'left'))

        # deltaL = hwUpL * self.upperRestL.inv()
        # dqL = deltaL.as_quat()
        # tareQ_arrL = np.array([0, dqL[1], 0, dqL[3]])
        # if np.linalg.norm(tareQ_arrL) < 0.0001:
        #     tareQ_arrL = np.array([0, 1, 0, 0])
        # tareQ_arrL = tareQ_arrL / np.linalg.norm(tareQ_arrL)
        # self.tareL = to_R(tareQ_arrL)
        eulerL = hwUpL.as_euler('YXZ',degrees = False)
        self.tareL = R.from_euler('XYZ',[0,eulerL[0],0],degrees = False)
        self.upperMountCorrL = hwUpL.inv() * self.tareL * self.upperRestWorldL
        self.forearmMountLL = self.upperMountCorrL.inv()
        self.forearmMountRL = hwFoL.inv() * self.upperMountCorrL * self.forearmRestL
        self.handMountLL = self.forearmMountRL.inv()
        self.handMountRL = hwHaL.inv() * self.forearmMountRL * self.handRestL

        self.is_calibrated = True

    def process_arm(self, r_imu, l_imu):
        hwUpR = to_R(ConvertToThreeSpace(r_imu['upperArm'], 'right'))
        hwFoR = to_R(ConvertToThreeSpace(r_imu['forearm'], 'right'))
        hwHaR = to_R(ConvertToThreeSpace(r_imu['hand'], 'right'))
        
        alUpR = self.tareR.inv() * hwUpR * self.upperMountCorrR
        alFoR = self.forearmMountLR * hwFoR * self.forearmMountRR
        alHaR = self.handMountLR * hwHaR * self.handMountRR

        hwUpL = to_R(ConvertToThreeSpace(l_imu['upperArm'], 'left'))
        hwFoL = to_R(ConvertToThreeSpace(l_imu['forearm'], 'left'))
        hwHaL = to_R(ConvertToThreeSpace(l_imu['hand'], 'left'))
        
        alUpL = self.tareL.inv() * hwUpL * self.upperMountCorrL
        alFoL = self.forearmMountLL * hwFoL * self.forearmMountRL
        alHaL = self.handMountLL * hwHaL * self.handMountRL

        return {
            'right': {'upperArm': alUpR, 'forearm': alFoR, 'hand': alHaR},
            'left': {'upperArm': alUpL, 'forearm': alFoL, 'hand': alHaL}
        }

def get_WXYZ(r: R):
    q = r.as_quat() # [x, y, z, w]
    return [q[3], q[0], q[1], q[2]]

def unpack_hand(data, offset):
    def unpack_quat(off):
        w = struct.unpack_from('<h', data, off + 0)[0] / 32767.0
        x = struct.unpack_from('<h', data, off + 2)[0] / 32767.0
        y = struct.unpack_from('<h', data, off + 4)[0] / 32767.0
        z = struct.unpack_from('<h', data, off + 6)[0] / 32767.0
        if w == 0 and x == 0 and y == 0 and z == 0:
            return [np.nan, np.nan, np.nan, np.nan], True
        q = np.array([x, y, z, w])
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm
        return q.tolist(), False

    rQ_U, u_is_zero = unpack_quat(offset + 0)
    rQ_F, f_is_zero = unpack_quat(offset + 8)
    rQ_H, h_is_zero = unpack_quat(offset + 16)
    
    fingers = []
    for f in range(5):
        fingers.extend(struct.unpack_from('<bbb', data, offset + 24 + f * 3))
    thumbExtra = struct.unpack_from('<b', data, offset + 39)[0]
    
    status = struct.unpack_from('<B', data, offset + 72)[0]
    calStatus = status & 0x1F
    connected = ((status >> 5) & 0x1) == 1
    
    imu_dict = {
        'upperArm': rQ_U,
        'forearm': rQ_F,
        'hand': rQ_H
    }
    
    finger_floats = list(fingers) + [thumbExtra]
    
    return imu_dict, finger_floats, connected, u_is_zero

class AppUI:
    def __init__(self, ip):
        self.ip = ip
        self.status = "Initializing..."
        self.calibrated = False
        self.calib_frames = 0
        self.prediction = "WAITING"
        self.confidence = 0.0
        self.all_probs = {}
        self.error = None
        self.history = []
        self.trigger_recalibrate = False
        self.debug_text = "Waiting for data stream..."
        self.last_detection_frame = 0
        self.is_inferring = False
        self.current_sequence = []
        self.last_emitted_times = {}
        self.fps = 0.0
        self.relay_logs = []

    def log_relay(self, msg):
        self.relay_logs.append(msg)
        if len(self.relay_logs) > 10:
            self.relay_logs.pop(0)

    def make_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="debug", size=13),
            Layout(name="footer", size=4)
        )
        layout["body"].split_row(
            Layout(name="main", ratio=2),
            Layout(name="history", ratio=1),
            Layout(name="relay_logs", ratio=1)
        )
        
        header_text = Text(f" ESL Glove Live Inference | Target: ws://{self.ip}:81 ", style="bold white on blue", justify="center")
        layout["header"].update(Panel(header_text, style="blue"))

        # Main content
        if self.error:
            main_content = Align.center(Text(f"ERROR: {self.error}", style="bold red", justify="center"), vertical="middle")
            self.error = None
        else:
            if self.prediction != "WAITING":
                pred_color = "green" if self.confidence > 0.6 else "yellow"
                main_text = Text(f"{fix_arabic(self.prediction)}\n", style=f"bold {pred_color}", justify="center")
                main_text.stylize(f"bold {pred_color}")
                main_text.append(f"\nConfidence: {self.confidence:.2f}", style="cyan")
                
                if self.all_probs:
                    prob_table = Table.grid(expand=False, padding=(0, 2))
                    prob_table.add_column(justify="right", style="cyan bold")
                    prob_table.add_column(justify="left", style="white")
                    
                    for label, prob in list(self.all_probs.items())[:5]:
                        prob_table.add_row(f"{prob*100:5.1f}%", fix_arabic(label.upper()))
                        
                    main_group = Group(
                        Align.center(main_text),
                        Text("\n[Live Probabilities]", style="dim", justify="center"),
                        Align.center(prob_table)
                    )
                    main_content = Align.center(main_group, vertical="middle")
                else:
                    main_content = Align.center(main_text, vertical="middle")
            else:
                main_group_items = [Align.center(Text("Waiting for signs...", style="dim", justify="center"))]
                if self.all_probs:
                    prob_table = Table.grid(expand=False, padding=(0, 2))
                    prob_table.add_column(justify="right", style="cyan bold")
                    prob_table.add_column(justify="left", style="white")
                    
                    for label, prob in list(self.all_probs.items())[:5]:
                        prob_table.add_row(f"{prob*100:5.1f}%", fix_arabic(label.upper()))
                            
                    main_group_items.extend([
                        Text("\n[Live Probabilities]", style="dim", justify="center"),
                        Align.center(prob_table)
                    ])
                main_content = Align.center(Group(*main_group_items), vertical="middle")
        
        layout["main"].update(Panel(main_content, title="Live Prediction", border_style="green" if self.calibrated else "yellow"))

        # History content
        history_text = Text()
        for h in reversed(self.history[-10:]):
            history_text.append(f"• {fix_arabic(h)}\n", style="white")
        layout["history"].update(Panel(history_text, title="History", border_style="blue"))
        
        # Relay Logs
        relay_text = Text()
        for r in self.relay_logs:
            relay_text.append(f"• {r}\n", style="cyan")
        layout["relay_logs"].update(Panel(relay_text, title="Relay Logs", border_style="cyan"))
        
        # Debug content
        debug_panel = Panel(Text(self.debug_text, style="dim white"), title="Raw Hardware Debug (Live)", border_style="cyan")
        layout["debug"].update(debug_panel)

        # Footer content
        cal_status = f"[green]Calibrated[/green]" if self.calibrated else f"[yellow]Needs Calibration (Press 'C')[/yellow]"
        status_text = f"Status: {self.status} | FPS: {self.fps:.1f} | Calibration: {cal_status}"
        cmds_text = "Commands: [bold cyan]C[/bold cyan] Force Recalibrate | [bold cyan]Q[/bold cyan] Quit"
        
        footer_table = Table.grid(expand=True)
        footer_table.add_column(justify="left", ratio=1)
        footer_table.add_column(justify="right", ratio=1)
        footer_table.add_row(status_text, cmds_text)
        
        layout["footer"].update(Panel(footer_table, border_style="blue"))
        return layout

async def glove_inference_client(ip="192.168.1.8"):
    uri = f"ws://{ip}:81"
    model_dir = os.path.dirname(os.path.abspath(__file__))
    
    ui = AppUI(ip)
    
    try:
        loop = asyncio.get_event_loop()
        recognizer = SlidingWindowRecognizer(model_dir=os.path.join(model_dir, "..", "models"))
    except Exception as e:
        ui.error = f"Error loading model: {e}"
        print(f"Error loading model: {e}")
        return

    calibrator = GloveCalibrator()
    frames_buffer = []
    timestamps_buffer = []
    max_window_size = max(recognizer.window_sizes)

    # Start the relay broadcast server
    async def relay_wrapper(*args, **kwargs):
        await _relay_handler(args[0], ui_ref=ui)
    relay_server = await websockets.serve(relay_wrapper, "0.0.0.0", RELAY_PORT)
    print(f"[Relay] Broadcast server listening on port {RELAY_PORT}")

    # Keyboard Listener Thread
    def keyboard_listener():
        while True:
            if msvcrt.kbhit():
                key = msvcrt.getch().decode('utf-8', 'ignore').lower()
                if key == 'q':
                    os._exit(0)
                elif key == 'c':
                    ui.trigger_recalibrate = True
            time.sleep(0.05)
    
    threading.Thread(target=keyboard_listener, daemon=True).start()

    with Live(ui.make_layout(), refresh_per_second=15, screen=True) as live:
        def update_ui():
            live.update(ui.make_layout())

        frame_counter = 0
        inference_cursor = 0
        new_data_event = asyncio.Event()

        def run_inference_task(t, v):
            _, preprocessed = preprocess_stream(t, v, disabled_groups=cfg.DISABLED_FEATURE_GROUPS)
            dets = recognizer.recognize(preprocessed)
            return dets, getattr(recognizer, 'latest_probs', {})

        async def inference_worker():
            nonlocal inference_cursor
            while True:
                await new_data_event.wait()
                
                while len(frames_buffer) - inference_cursor >= max_window_size + 10:
                    ui.is_inferring = True
                    try:
                        target_len = max_window_size + 10
                        t_slice = timestamps_buffer[inference_cursor : inference_cursor + target_len]
                        v_slice = frames_buffer[inference_cursor : inference_cursor + target_len]
                        
                        f_counter = frame_counter - len(frames_buffer) + inference_cursor + target_len
                        
                        t_arr = np.array(t_slice)
                        v_arr = np.array(v_slice)
                        
                        detections, live_probs = await asyncio.to_thread(run_inference_task, t_arr, v_arr)
                        
                        ui.all_probs = live_probs
                        
                        if detections:
                            for d in detections:
                                absolute_end = f_counter - len(v_arr) + d['end']
                                word = d['label'].upper()
                                
                                is_duplicate = False
                                if ui.current_sequence and ui.current_sequence[-1] == word:
                                    is_duplicate = True
                                elif word in ui.last_emitted_times:
                                    if absolute_end - ui.last_emitted_times[word] < 250:
                                        is_duplicate = True
                                        
                                if not is_duplicate:
                                    ui.last_emitted_times[word] = absolute_end
                                    
                                    if frame_counter - ui.last_detection_frame > 250:
                                        ui.current_sequence = []
                                        
                                    ui.current_sequence.append(word)
                                    ui.prediction = " ".join(ui.current_sequence)
                                    ui.confidence = d['confidence']
                                    ui.last_detection_frame = frame_counter
                                    
                                    # Broadcast to relay clients (mobile app)
                                    sentence = ui.prediction
                                    t = asyncio.ensure_future(_broadcast(sentence, ui))
                                    _bg_tasks.add(t)
                                    t.add_done_callback(lambda task, u=ui: _task_done(task, u))
                                    
                                    if not ui.history or ui.history[-1] != word:
                                        ui.history.append(word)
                                            
                        update_ui()
                    except Exception as e:
                        ui.debug_text = f"Inference Error: {type(e).__name__} - {str(e)}"
                        update_ui()
                    finally:
                        ui.is_inferring = False
                    
                    inference_cursor += 10
                    
                    # Catch up if we're falling too far behind to prevent lag
                    if len(frames_buffer) - inference_cursor > max_window_size + 50:
                        inference_cursor = len(frames_buffer) - max_window_size - 10

                new_data_event.clear()

        worker_task = asyncio.create_task(inference_worker())

        fps_frame_count = 0
        fps_start_time = time.time()

        while True:
            try:
                async with websockets.connect(uri, ping_interval=None) as websocket:
                    ui.status = "[green]Connected[/green]"
                    update_ui()
                    
                    while True:
                        if ui.trigger_recalibrate:
                            calibrator.is_calibrated = False
                            ui.calibrated = False
                            ui.calib_frames = 0
                            ui.status = "[yellow]Recalibrating...[/yellow]"
                            ui.trigger_recalibrate = False
                            update_ui()

                        message = await websocket.recv()
                        if isinstance(message, str) or len(message) < 154:
                            continue
                            
                        header = struct.unpack_from('<I', message, 0)[0]
                        if header != 0x45534C47:
                            continue
                            
                        timestamp_us = struct.unpack_from('<I', message, 4)[0]
                            
                        right_imu, right_fingers, right_conn, r_u_zero = unpack_hand(message, 8)
                        left_imu, left_fingers, left_conn, l_u_zero = unpack_hand(message, 81)

                        if r_u_zero or np.isnan(right_imu['upperArm'][0]):
                            continue

                        frame_counter += 1
                        fps_frame_count += 1
                        
                        now = time.time()
                        if now - fps_start_time >= 1.0:
                            ui.fps = fps_frame_count / (now - fps_start_time)
                            fps_start_time = now
                            fps_frame_count = 0
                        
                        if frame_counter % 5 == 0:
                            def fmt_q(q): return f"[{q[0]:.2f}, {q[1]:.2f}, {q[2]:.2f}, {q[3]:.2f}]"
                            ui.debug_text = (
                                f"[R_IMU] U: {fmt_q(right_imu['upperArm'])} | F: {fmt_q(right_imu['forearm'])} | H: {fmt_q(right_imu['hand'])}\n"
                                f"[L_IMU] U: {fmt_q(left_imu['upperArm'])} | F: {fmt_q(left_imu['forearm'])} | H: {fmt_q(left_imu['hand'])}\n"
                                f"[R_FINGERS] {right_fingers}\n"
                                f"[L_FINGERS] {left_fingers}"
                            )
                            update_ui()

                        if not calibrator.is_calibrated:
                            ui.calib_frames += 1
                            if ui.calib_frames > 30:
                                calibrator.calibrate(right_imu, left_imu)
                                ui.calibrated = True
                                ui.status = "[green]Calibrated[/green]"
                                update_ui()
                            else:
                                ui.status = f"[yellow]Aligning... {ui.calib_frames}/30[/yellow]"
                                update_ui()
                            continue
                            
                        # Process Frame
                        cal_imus = calibrator.process_arm(right_imu, left_imu)
                        rPalm = cal_imus['right']
                        lPalm = cal_imus['left']
                        
                        flat56 = []
                        flat56.extend(right_fingers)
                        flat56.extend(get_WXYZ(rPalm['hand']))
                        flat56.extend(get_WXYZ(rPalm['forearm']))
                        flat56.extend(get_WXYZ(rPalm['upperArm']))
                        
                        flat56.extend(left_fingers)
                        flat56.extend(get_WXYZ(lPalm['hand']))
                        flat56.extend(get_WXYZ(lPalm['forearm']))
                        flat56.extend(get_WXYZ(lPalm['upperArm']))
                        
                        frames_buffer.append(flat56)
                        timestamps_buffer.append(frame_counter * 20000.0)  # Simulated timestamps at 50Hz (20ms) to guarantee monotonicity
                        
                        if len(frames_buffer) > 600:
                            frames_buffer.pop(0)
                            timestamps_buffer.pop(0)
                            inference_cursor = max(0, inference_cursor - 1)
                            
                        if len(frames_buffer) - inference_cursor >= max_window_size + 10:
                            new_data_event.set()
                                        
                        # Clear old predictions after 5 seconds (250 frames)
                        if len(frames_buffer) - inference_cursor < max_window_size + 10 and not ui.is_inferring:
                            if frame_counter - ui.last_detection_frame > 250 and ui.prediction != "WAITING":
                                ui.prediction = "WAITING"
                                ui.current_sequence = []
                                ui.confidence = 0.0
                                ui.all_probs = {}
                                update_ui()
                                t = asyncio.ensure_future(_broadcast("WAITING", ui))
                                _bg_tasks.add(t)
                                t.add_done_callback(lambda task, u=ui: _task_done(task, u))

            except Exception as e:
                ui.error = f"Connection Lost. Retrying... (Error: {type(e).__name__} - {str(e)})"
                ui.status = "[red]Disconnected[/red]"
                update_ui()
                await asyncio.sleep(2)
                ui.error = None

if __name__ == "__main__":
    ip_address = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.8"
    asyncio.run(glove_inference_client(ip_address))
