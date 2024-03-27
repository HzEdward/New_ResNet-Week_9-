import matplotlib.pyplot as plt

# 读取文本文件
file_path = "./loss_record_trial.txt"  # 替换为您的文本文件路径
batch_numbers = []
loss_values = []

with open(file_path, 'r') as file:
    for line in file:
        if line.startswith("Epoch"):
            parts = line.split(" - ")
            batch_loss = parts[1].split(": ")[1]
            loss_values.append(float(batch_loss))
            batch_number = parts[0].split(", Batch ")[1]
            batch_numbers.append(int(batch_number))

# 绘制图表
plt.figure(figsize=(10, 5))
plt.plot(batch_numbers, loss_values, marker='o', linestyle='-')
plt.title('Loss vs. Batch Number')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
