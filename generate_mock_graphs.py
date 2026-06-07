import matplotlib.pyplot as plt
import os

base_dir = r"c:\Dev\Graduation_Project\GP_CodeSubmission\ESL-glove-model-development"

# 1. Class Accuracy Graph
labels_list = ["Hello", "Thank You", "Yes", "No", "Please", "Sorry", "Help", "Good", "Bad", "You"]
accuracy_list = [95.5, 92.0, 98.2, 97.5, 88.0, 89.5, 91.0, 94.5, 93.0, 96.0]

plt.figure(figsize=(14, 7))
plt.bar(labels_list, accuracy_list, color='skyblue')
plt.xlabel('Sign Class')
plt.ylabel('Accuracy (%)')
plt.title('Class-wise Recognition Accuracy (Mock Data)')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 100)
for i, v in enumerate(accuracy_list):
    plt.text(i, v + 1, str(v)+'%', ha='center', va='bottom')
plt.tight_layout()
plt.savefig(os.path.join(base_dir, 'class_accuracy_graph.png'))
plt.close()

# 2. Sequence Accuracy Graph
correct = 85
incorrect = 15

plt.figure(figsize=(6, 6))
plt.pie([correct, incorrect], labels=['Correct Sequences', 'Incorrect Sequences'], autopct='%1.1f%%', colors=['#4CAF50', '#F44336'])
plt.title('Sequence Exact-Match Accuracy (Mock Data: 85.0%)')
plt.savefig(os.path.join(base_dir, 'sequence_accuracy_graph.png'))
plt.close()

print("Mock graphs generated.")
