"""Rank actual weighted TensorBoard penalty curves; never rank coefficients alone."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    events = EventAccumulator(str(args.run), size_guidance={'scalars': 0})
    events.Reload()
    tags = [t for t in events.Tags()['scalars'] if t.startswith('Episode_Reward/')]
    end = max(e.step for t in tags for e in events.Scalars(t))
    rows, curves = [], {}
    for tag in tags:
        samples = events.Scalars(tag)
        steps, values = np.array([e.step for e in samples]), np.array([e.value for e in samples])
        window = values[steps > end-1000]
        near26 = values[(steps >= 25500) & (steps <= 26500)]
        name = tag.split('/')[-1]
        curves[name] = (steps, values)
        rows.append(dict(name=name, last_mean=float(window.mean()), last_p05=float(np.quantile(window,.05)),
                         last_min=float(window.min()), last_max=float(window.max()), samples=len(window),
                         near26k_mean=float(near26.mean()) if len(near26) else None))
    penalties = sorted([r for r in rows if r['last_mean'] < 0], key=lambda r:r['last_mean'])
    total = -sum(r['last_mean'] for r in penalties)
    for row in penalties:
        row['share_of_negative_means'] = -row['last_mean']/total
    data = dict(run=str(args.run.resolve()), final_step=int(end), window=[int(end-999),int(end)],
                method='TensorBoard Episode_Reward: weighted episode sum divided by configured maximum episode seconds. '
                       'Means are over logged training updates, not equal-weight physical transitions or real-world evaluations. '
                       'Shares measure logged penalty magnitude, not gradient size or causal responsibility.',
                penalties=penalties, all_terms=rows)
    (args.output/'ranking.json').write_text(json.dumps(data,indent=2)+'\n')
    np.savez_compressed(args.output/'curves.npz', **{n+'__'+k:arr for n,pair in curves.items() for k,arr in zip(('step','value'),pair)})
    fig, axes = plt.subplots(2,1,figsize=(12,9),constrained_layout=True)
    top = penalties[:10]
    axes[0].barh([r['name'] for r in top][::-1], [-r['last_mean'] for r in top][::-1],color='#326c9d')
    axes[0].set_xlabel('Mean weighted penalty magnitude (last 1,000 updates)')
    axes[0].set_title('ELF3: observed penalties, not configured weights')
    for row in penalties[:5]:
        steps, values=curves[row['name']]
        edges=np.arange(0,len(values),200)
        xs=[steps[i:i+200].mean() for i in edges]
        ys=[-values[i:i+200].mean() for i in edges]
        axes[1].plot(xs,ys,label=row['name'])
    axes[1].set_xlabel('Training iteration');axes[1].set_ylabel('Weighted penalty magnitude')
    axes[1].legend(fontsize=8,ncol=2);axes[1].grid(alpha=.2)
    fig.savefig(args.output/'penalty_curves.png',dpi=170);plt.close(fig)
    lines=['# 实际惩罚曲线排名','',f'训练目录：`{args.run.name}`。末段窗口：{end-999}–{end}。','',
           '这里比较 TensorBoard 中已经乘权重的回合分项值，再按配置的最大回合秒数归一化。它不是原始物理误差，也不能直接当作每秒扣分、梯度大小或问题的因果归因。','',
           '| 项目 | 26k 附近均值 | 最后 1000 轮均值 | 负项总量占比 | 最后窗口最差单点 |',
           '|---|---:|---:|---:|---:|']
    for row in penalties:
        near='—' if row['near26k_mean'] is None else f"{row['near26k_mean']:.6f}"
        lines.append(f"| {row['name']} | {near} | {row['last_mean']:.6f} | {row['share_of_negative_means']:.1%} | {row['last_min']:.6f} |")
    lines += ['', '![实际扣分及趋势](penalty_curves.png)', '',
              '非足部接触项统计除 ankle 外的身体接触；仅凭这条曲线不能确定是哪一身体部位或哪种碰撞。',
              '终止惩罚和力矩超限在本次 Isaac 训练曲线里均值较小，不代表 MuJoCo 或实机上的跌倒/力矩风险低。',
              '训练随机策略、命令混合比例和扰动设置会影响曲线；定点部署评测仍需单独判断。','']
    (args.output/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(top5=penalties[:5],negative_total=total),indent=2))


if __name__ == '__main__':
    main()
