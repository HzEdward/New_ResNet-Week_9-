import re
import matplotlib.pyplot as plt

def calculate_average_loss(loss_values, step):
    average_losses = []

    for i in range(0, len(loss_values), step):
        batch_losses = loss_values[i:i+step]
        average_loss = sum(batch_losses) / len(batch_losses)
        average_losses.append(average_loss)

    return average_losses

def plot_loss(file_path, save_path, step=1):
    batch_numbers = []
    loss_values = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Epoch"):
                parts = line.split(" - ")
                batch_loss = re.search(r'Loss: (\d+\.\d+)', parts[1]).group(1)
                loss_values.append(float(batch_loss))
                batch_number = parts[0].split(", Batch ")[1]
                batch_numbers.append(int(batch_number))

    if step > 1:
        loss_values = calculate_average_loss(loss_values, step)
        batch_numbers = list(range(step, len(loss_values)*step + step, step))

    # 绘制图表
    plt.figure(figsize=(10, 5))
    plt.plot(batch_numbers, loss_values, marker='o', linestyle='-')
    plt.title('Loss vs. Batch Number')
    plt.xlabel('Batch Number')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)  # 保存图表为图片

if __name__ == "__main__":
    file_path = "./loss_record_trial.txt"  # 替换为您的文本文件路径
    save_path = "loss_plot.png"  # 图片保存路径
    plot_loss(file_path, save_path, step=50)  # 每隔50个批次显示一个数据点
