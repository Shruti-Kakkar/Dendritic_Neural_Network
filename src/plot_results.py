import matplotlib.pyplot as plt
import numpy as np
import os

# ── all results collected from your runs ─────────────────────

results = {
    'vANN': {
        'test_acc':  [45.22, 44.69, 45.05],
        'test_loss': [2.2789, 2.3615, 2.3026],
        'train_loss_curves': [
            [1.8757,1.6795,1.5650,1.4778,1.3967,1.3347,1.2661,1.1952,1.1302,1.0608,
             0.9958,0.9275,0.8620,0.7992,0.7359,0.6741,0.6225,0.5641,0.5220,0.4790],
            [1.8723,1.6719,1.5592,1.4729,1.3996,1.3289,1.2619,1.1965,1.1307,1.0612,
             0.9978,0.9307,0.8599,0.8060,0.7351,0.6812,0.6235,0.5621,0.5254,0.4809],
            [1.8782,1.6783,1.5672,1.4860,1.4064,1.3367,1.2683,1.2004,1.1365,1.0648,
             0.9952,0.9350,0.8651,0.8016,0.7373,0.6748,0.6199,0.5715,0.5173,0.4717],
        ],
        'val_loss_curves': [
            [1.7721,1.6547,1.6204,1.5915,1.6180,1.5989,1.5811,1.5855,1.6214,1.6376,
             1.6964,1.7376,1.8383,1.8336,1.9006,2.0009,2.0164,2.1549,2.2607,2.3394],
            [1.7659,1.6783,1.6494,1.6038,1.6102,1.5943,1.6255,1.6323,1.6590,1.6866,
             1.7167,1.7855,1.8135,1.8940,1.9804,2.0683,2.1111,2.2277,2.3253,2.4413],
            [1.7604,1.6700,1.6261,1.5820,1.5737,1.5696,1.5587,1.6071,1.6331,1.6661,
             1.6698,1.7385,1.7583,1.8360,1.8617,1.9629,2.0238,2.1014,2.2041,2.2614],
        ],
        'color': '#e74c3c',
        'params': 399114,
    },
    'dANN-R': {
        'test_acc':  [47.96, 48.06, 48.19],
        'test_loss': [1.7065, 1.6968, 1.6783],
        'train_loss_curves': [
            [1.9505,1.7040,1.5949,1.5135,1.4409,1.3722,1.3059,1.2458,1.1839,1.1252,
             1.0674,1.0079,0.9517,0.8997,0.8445,0.7902,0.7421,0.6910,0.6405,0.5977],
            [1.9500,1.7036,1.5952,1.5120,1.4386,1.3706,1.3079,1.2444,1.1842,1.1263,
             1.0676,1.0106,0.9498,0.8960,0.8388,0.7901,0.7374,0.6922,0.6402,0.5961],
            [1.9516,1.7045,1.5999,1.5183,1.4482,1.3825,1.3201,1.2604,1.2023,1.1379,
             1.0845,1.0240,0.9650,0.9159,0.8591,0.8076,0.7548,0.7059,0.6575,0.6124],
        ],
        'val_loss_curves': [
            [1.7916,1.6811,1.6249,1.5772,1.5467,1.5424,1.5133,1.5013,1.5125,1.5266,
             1.5334,1.5244,1.5314,1.5602,1.5658,1.5917,1.6206,1.6458,1.6553,1.7005],
            [1.7973,1.6923,1.6294,1.5759,1.5643,1.5640,1.5311,1.5086,1.5213,1.5177,
             1.5480,1.5349,1.5375,1.5698,1.5873,1.5873,1.6173,1.6518,1.7050,1.7291],
            [1.7757,1.6853,1.5972,1.5690,1.5302,1.5102,1.5006,1.4904,1.4810,1.4776,
             1.4791,1.4848,1.5067,1.5084,1.5213,1.5403,1.5672,1.6025,1.6228,1.6567],
        ],
        'color': '#3498db',
        'params': 399114,
    },
    'dANN-LRF-4x4': {
        'test_acc':  [53.31, 53.52, 52.19],
        'test_loss': [1.3959, 1.4109, 1.4140],
        'train_loss_curves': [
            [1.9706,1.6787,1.5351,1.4513,1.3839,1.3312,1.2832,1.2382,1.1968,1.1550,
             1.1166,1.0787,1.0428,1.0099,0.9747,0.9429,0.9114,0.8790,0.8508,0.8229],
            [1.9651,1.6724,1.5358,1.4513,1.3874,1.3386,1.2906,1.2461,1.2038,1.1610,
             1.1245,1.0905,1.0524,1.0158,0.9846,0.9507,0.9200,0.8913,0.8612,0.8300],
            [1.9653,1.6764,1.5356,1.4505,1.3894,1.3353,1.2868,1.2439,1.2001,1.1599,
             1.1222,1.0840,1.0513,1.0116,0.9820,0.9491,0.9157,0.8873,0.8592,0.8278],
        ],
        'val_loss_curves': [
            [1.8100,1.6308,1.5463,1.4857,1.4650,1.4431,1.4265,1.4116,1.4073,1.3990,
             1.3867,1.3649,1.3836,1.3830,1.3950,1.4062,1.3921,1.4004,1.4238,1.4185],
            [1.7901,1.6313,1.5624,1.5061,1.4679,1.4427,1.4346,1.4285,1.4121,1.3989,
             1.3874,1.4037,1.3906,1.3909,1.3835,1.3797,1.4108,1.4159,1.4196,1.4190],
            [1.7954,1.6257,1.5375,1.4823,1.4566,1.4254,1.4027,1.4058,1.3979,1.3699,
             1.3804,1.3473,1.3550,1.3617,1.3642,1.3562,1.3964,1.3777,1.3763,1.3944],
        ],
        'color': '#2ecc71',
        'params': 399114,
    },
    'dANN-LRF-4x8': {
        'test_acc':  [52.64, 51.64, 53.37],
        'test_loss': [1.4436, 1.4609, 1.4249],
        'train_loss_curves': [
            [1.9524,1.6432,1.5260,1.4506,1.3881,1.3278,1.2789,1.2297,1.1852,1.1415,
             1.0987,1.0598,1.0249,0.9879,0.9535,0.9216,0.8865,0.8583,0.8266,0.7985],
            [1.9529,1.6414,1.5274,1.4508,1.3884,1.3313,1.2790,1.2299,1.1836,1.1427,
             1.1016,1.0625,1.0248,0.9896,0.9510,0.9191,0.8879,0.8541,0.8218,0.7903],
            [1.9597,1.6456,1.5255,1.4516,1.3891,1.3321,1.2830,1.2345,1.1911,1.1501,
             1.1042,1.0674,1.0320,0.9963,0.9586,0.9263,0.8931,0.8614,0.8334,0.7995],
        ],
        'val_loss_curves': [
            [1.7636,1.6173,1.5355,1.4998,1.4692,1.4459,1.4323,1.4138,1.4086,1.3864,
             1.3984,1.3871,1.3967,1.3977,1.4097,1.4000,1.4211,1.4250,1.4407,1.4767],
            [1.7535,1.6101,1.5582,1.5201,1.4848,1.4685,1.4383,1.4210,1.4116,1.4137,
             1.4169,1.3960,1.4200,1.4186,1.4123,1.4279,1.4284,1.4401,1.4754,1.4785],
            [1.7494,1.5940,1.5346,1.4907,1.4595,1.4274,1.4123,1.4015,1.3743,1.3705,
             1.3627,1.3594,1.3566,1.3511,1.3522,1.3685,1.3585,1.3737,1.3790,1.3939],
        ],
        'color': '#9b59b6',
        'params': 403210,
    },
}

epochs = list(range(1, 21))

# ── helper: mean and std across seeds ────────────────────────
def mean_std(curves):
    arr = np.array(curves)
    return arr.mean(axis=0), arr.std(axis=0)

os.makedirs('./outputs', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Plot 1 — Test Accuracy Bar Chart
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

names  = list(results.keys())
means  = [np.mean(results[m]['test_acc'])  for m in names]
stds   = [np.std(results[m]['test_acc'])   for m in names]
colors = [results[m]['color']              for m in names]

bars = ax.bar(names, means, yerr=stds, capsize=6,
              color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

# annotate bars
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + std + 0.3,
            f'{mean:.2f}%', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title('Test Accuracy Comparison\n(parameter-matched, CIFAR-10 grayscale)',
             fontsize=13)
ax.set_ylim(40, 60)
ax.axhline(y=np.mean(results['vANN']['test_acc']),
           color='#e74c3c', linestyle='--', alpha=0.5, label='vANN baseline')
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./outputs/plot1_accuracy_comparison.png', dpi=150)
plt.close()
print("Saved plot1_accuracy_comparison.png")


# ─────────────────────────────────────────────────────────────
# Plot 2 — Overfitting Curves (mean across seeds)
# ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for name, data in results.items():
    train_mean, train_std = mean_std(data['train_loss_curves'])
    val_mean,   val_std   = mean_std(data['val_loss_curves'])
    c = data['color']

    # train loss
    axes[0].plot(epochs, train_mean, color=c, label=name, linewidth=2)
    axes[0].fill_between(epochs,
                          train_mean - train_std,
                          train_mean + train_std,
                          color=c, alpha=0.15)

    # val loss
    axes[1].plot(epochs, val_mean, color=c, label=name, linewidth=2)
    axes[1].fill_between(epochs,
                          val_mean - val_std,
                          val_mean + val_std,
                          color=c, alpha=0.15)

axes[0].set_title('Training Loss', fontsize=13)
axes[0].set_xlabel('Epoch', fontsize=11)
axes[0].set_ylabel('Loss', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

axes[1].set_title('Validation Loss', fontsize=13)
axes[1].set_xlabel('Epoch', fontsize=11)
axes[1].set_ylabel('Loss', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(alpha=0.3)

fig.suptitle('Training vs Validation Loss\n(mean ± std across 3 seeds)',
             fontsize=13)
plt.tight_layout()
plt.savefig('./outputs/plot2_overfitting_curves.png', dpi=150)
plt.close()
print("Saved plot2_overfitting_curves.png")


# ─────────────────────────────────────────────────────────────
# Plot 3 — Overfitting Gap Bar Chart
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))

gaps   = []
g_stds = []
for name, data in results.items():
    train_ep20 = [c[-1] for c in data['train_loss_curves']]
    val_ep20   = [c[-1] for c in data['val_loss_curves']]
    gap_per_seed = [v - t for v, t in zip(val_ep20, train_ep20)]
    gaps.append(np.mean(gap_per_seed))
    g_stds.append(np.std(gap_per_seed))

bars = ax.bar(names, gaps, yerr=g_stds, capsize=6,
              color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)

for bar, gap, std in zip(bars, gaps, g_stds):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + std + 0.01,
            f'{gap:.2f}', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Val Loss − Train Loss at Epoch 20', fontsize=11)
ax.set_title('Overfitting Gap at Epoch 20\n(smaller is better)', fontsize=13)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('./outputs/plot3_overfitting_gap.png', dpi=150)
plt.close()
print("Saved plot3_overfitting_gap.png")


# ─────────────────────────────────────────────────────────────
# Plot 4 — Summary Table Plot
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis('off')

table_data = []
for name, data in results.items():
    acc_mean  = np.mean(data['test_acc'])
    acc_std   = np.std(data['test_acc'])
    loss_mean = np.mean(data['test_loss'])
    loss_std  = np.std(data['test_loss'])
    train_ep20 = np.mean([c[-1] for c in data['train_loss_curves']])
    val_ep20   = np.mean([c[-1] for c in data['val_loss_curves']])
    gap        = val_ep20 - train_ep20
    table_data.append([
        name,
        f"{data['params']:,}",
        f"{acc_mean:.2f} ± {acc_std:.2f}",
        f"{loss_mean:.4f} ± {loss_std:.4f}",
        f"{gap:.2f}"
    ])

columns = ['Model', 'Parameters', 'Test Acc (%)', 'Test Loss', 'Overfit Gap']
table = ax.table(cellText=table_data, colLabels=columns,
                 cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 2.0)

# highlight header
for j in range(len(columns)):
    table[0, j].set_facecolor('#2c3e50')
    table[0, j].set_text_props(color='white', fontweight='bold')

# highlight best accuracy row (dANN-LRF-4x4, index 2)
for j in range(len(columns)):
    table[3, j].set_facecolor('#d5f5e3')

ax.set_title('Results Summary — CIFAR-10 Grayscale',
             fontsize=13, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('./outputs/plot4_summary_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved plot4_summary_table.png")

print("\nAll plots saved to ./outputs/")