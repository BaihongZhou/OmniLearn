import numpy as np
from concurrent.futures import ThreadPoolExecutor
import os
from tqdm import tqdm

# 映射事件类型的字典
name_map = {
    "pi_pi": 0,
    "pi_rho": 1,
    "lep_pi": 2,
    "lep_rho": 3,
    "QCD": 4,
    "rho_rho": 5,
    "tt": 6,
    "Wlnu": 7,
    "Wtaunu": 8,
    "Zll": 9
}
number_to_name = {v: k for k, v in name_map.items()}

# 读取数据文件路径
raw_path = "/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/pi_pi/systematics/variation.13/merged_eval.npz"
raw_data = np.load(raw_path)

# 预先为每个事件类型的数据分配内存
event_dics = {name: {key: [] for key in raw_data.keys() if key != "Eevent_type"} for name in number_to_name.values()}

# 处理单个事件的函数
def process_event(i):
    event_name = number_to_name[raw_data["Eevent_type"][i]]
    for key in raw_data.keys():
        if key == "Eevent_type":
            continue
        # 将数据添加到对应事件类型的列表中
        event_dics[event_name][key].append(raw_data[key][i])

# 创建进度条对象
with tqdm(total=len(raw_data["Eevent_type"]), desc="Processing Events") as progress_bar:
    # 使用多线程并行化数据处理
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 提交每个任务，并手动更新进度条
        futures = []
        for i in range(len(raw_data["Eevent_type"])):
            future = executor.submit(process_event, i)
            futures.append(future)

        # 等待每个任务完成并更新进度条
        for future in futures:
            future.result()  # 等待任务完成
            progress_bar.update(1)  # 更新进度条

# 将列表转换为 numpy 数组并批量保存
def save_event_data(event_name):
    event_data = {key: np.stack(value) for key, value in event_dics[event_name].items()}
    np.savez(f"/global/cfs/cdirs/m2616/avencast/Quantum_Entanglement/workspace/results/pi_pi/systematics/variation.13/{event_name}_particle_eval.npz", **event_data)
    print(f"Saved {event_name} data to {event_name}_particle_eval.npz")

# 创建进度条对象
with tqdm(total=len(number_to_name), desc="Saving Data") as progress_bar:
    # 使用线程池并行化文件保存操作
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        # 提交每个保存任务，并手动更新进度条
        futures = []
        for event_name in number_to_name.values():
            future = executor.submit(save_event_data, event_name)
            futures.append(future)

        # 等待每个保存任务完成并更新进度条
        for future in futures:
            future.result()  # 等待任务完成
            progress_bar.update(1)  # 更新进度条
