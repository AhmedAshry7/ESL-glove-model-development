`import asyncio
import websockets
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
import sys
import os
import threading
import time
import msvcrt

from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.align import Align
from rich.text import Text
from rich.table import Table

# Ensure the sliding_window_recognizer can be imported
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from sliding_window_recognizer import SlidingWindowRecognizer

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

        self.upperRestL = to_R(REST_POSES["B-upperArm.L"]["local"])
        self.upperRestWorldL = to_R(REST_POSES["B-upperArm.L"]["world"])
        self.forearmRestL = to_R(REST_POSES["B-forearm.L"]["local"])
        self.handRestL = to_R(REST_POSES["B-hand.L"]["local"])

    def calibrate(self, r_imu, l_imu):
        # Right Arm
        hwUpR = to_R(ConvertToThreeSpace(r_imu['upperArm'], 'right'))
        hwFoR = to_R(ConvertToThreeSpace(r_imu['forearm'], 'right'))
        hwHaR = to_R(ConvertToThreeSpace(r_imu['hand'], 'right'))

        deltaR = hwUpR * self.upperRestR.inv()
        dqR = deltaR.as_quat()
        tareQ_arrR = np.array([0, dqR[1], 0, dqR[3]])
        if np.linalg.norm(tareQ_arrR) < 0.0001:
            tareQ_arrR = np.array([0, 1, 0, 0])
        tareQ_arrR = tareQ_arrR / np.linalg.norm(tareQ_arrR)
        self.tareR = to_R(tareQ_arrR)

        self.upperMountCorrR = hwUpR.inv() * self.tareR * self.upperRestWorldR
        self.forearmMountLR = self.upperMountCorrR.inv()
        self.forearmMountRR = hwFoR.inv() * self.upperMountCorrR * self.forearmRestR
        self.handMountLR = self.forearmMountRR.inv()
        self.handMountRR = hwHaR.inv() * self.forearmMountRR * self.handRestR

        # Left Arm
        hwUpL = to_R(ConvertToThreeSpace(l_imu['upperArm'], 'left'))
        hwFoL = to_R(ConvertToThreeSpace(l_imu['forearm'], 'left'))
        hwHaL = to_R(ConvertToThreeSpace(l_imu['hand'], 'left'))

        deltaL = hwUpL * self.upperRestL.inv()
        dqL = deltaL.as_quat()
        tareQ_arrL = np.array([0, dqL[1], 0, dqL[3]])
        if np.linalg.norm(tareQ_arrL) < 0.0001:
            tareQ_arrL = np.array([0, 1, 0, 0])
        tareQ_arrL = tareQ_arrL / np.linalg.norm(tareQ_arrL)
        self.tareL = to_R(tareQ_arrL)

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
        self.required_frames = 50
        self.prediction = "WAITING"
        self.confidence = 0.0
        self.error = None
        self.history = []
        self.trigger_recalibrate = False

    def make_layout(self):
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=4)
        )
        
        header_text = Text(f" ESL Glove Live Inference | Target: ws://{self.ip}:81 ", style="bold white on blue", justify="center")
        layout["header"].update(Panel(header_text, style="blue"))

        # Main content
        if self.error:
            main_content = Align.center(Text(f"ERROR: {self.error}", style="bold red", justify="center"), vertical="middle")
        else:
            if self.prediction != "WAITING":
                pred_color = "green" if self.confidence > 0.6 else "yellow"
                main_text = Text(f"{self.prediction}\n", style=f"bold {pred_color}", justify="center")
                main_text.stylize(f"bold {pred_color}")
                main_text.append(f"\nConfidence: {self.confidence:.2f}", style="cyan")
                
                # Make the prediction text large by using ASCII art or just bold large font, 
                # but standard rich doesn't have "large font" without pyfiglet. We'll use bold and center.
                main_content = Align.center(main_text, vertical="middle")
            else:
                main_content = Align.center(Text("Waiting for signs...", style="dim", justify="center"), vertical="middle")
        
        layout["main"].update(Panel(main_content, title="Live Prediction", border_style="green" if self.calibrated else "yellow"))

        # Footer content
        cal_status = f"[green]Calibrated[/green]" if self.calibrated else f"[yellow]Calibrating... ({self.calib_frames}/{self.required_frames})[/yellow]"
        status_text = f"Status: {self.status} | Calibration: {cal_status}"
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
        recognizer = SlidingWindowRecognizer(model_dir=os.path.join(model_dir, "..", "models"))
    except Exception as e:
        ui.error = f"Error loading model: {e}"
        print(f"Error loading model: {e}")
        return

    calibrator = GloveCalibrator()
    frames_buffer = []
    max_window_size = max(recognizer.window_sizes)

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

        while True:
            try:
                async with websockets.connect(uri) as websocket:
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
                            
                        right_imu, right_fingers, right_conn, r_u_zero = unpack_hand(message, 8)
                        left_imu, left_fingers, left_conn, l_u_zero = unpack_hand(message, 81)

                        if r_u_zero or np.isnan(right_imu['upperArm'][0]):
                            continue

                        # Auto-calibrate
                        if not calibrator.is_calibrated:
                            ui.calib_frames += 1
                            if ui.calib_frames >= ui.required_frames:
                                calibrator.calibrate(right_imu, left_imu)
                                ui.calibrated = True
                                ui.status = "[green]Running Inference[/green]"
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
                        
                        if len(frames_buffer) > max_window_size + 10:
                            frames_buffer.pop(0)
                            
                        if len(frames_buffer) >= max_window_size:
                            if len(frames_buffer) % 10 == 0:
                                arr = np.array(frames_buffer)
                                detections = recognizer.recognize(arr)
                                if detections:
                                    latest = detections[-1]
                                    ui.prediction = latest['label'].upper()
                                    ui.confidence = latest['confidence']
                                    update_ui()

            except Exception as e:
                ui.error = f"Connection Lost. Retrying..."
                ui.status = "[red]Disconnected[/red]"
                update_ui()
                await asyncio.sleep(2)
                ui.error = None

if __name__ == "__main__":
    ip_address = sys.argv[1] if len(sys.argv) > 1 else "192.168.1.8"
    asyncio.run(glove_inference_client(ip_address))
