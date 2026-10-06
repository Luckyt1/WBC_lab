"""Render paired, physical-outcome comparisons from a completed ablation suite."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory');args=p.parse_args()
    root=Path(args.directory);s=json.loads((root/'state.json').read_text())
    labels={'baseline':'原配置','knee':'仅横向膝间距','guidance':'仅起步引导组合','exploration':'仅髋探索下限','combined':'三项组合'}
    lines=['# 起步对照实验结果','',f"运行状态：{s['status']}。每组 {s['num_envs']} 环境、{s['updates_per_case']} 次更新，从相同第 2000 轮检查点及优化器开始。",'',
           '表中速度为固定指令评估的实际平均值；前进/后退目标为 +0.3 / -0.3 m/s，左右转向为 +0.5 / -0.5 rad/s。每项 32 次，4 秒过渡、12 秒测量。分数为四项跟踪误差按预定门槛归一化后的平均值，越低越好。','']
    for seed in sorted(set(r['seed'] for r in s['results'])):
        results=[r for r in s['results'] if r['seed']==seed]
        lines += [f'## 训练种子 {seed}','',
                  '| 组别 | 前进 vx | 后退 vx | 正转 wz | 反转 wz | 四项运动存活 /128 | 站立存活 /32 | 误差分数 | 基础任务全通过 |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|---|']
        for r in results:
            rows=r['scenarios'];nums=[]
            for n,a in [('forward',0),('backward',0),('turn_pos',2),('turn_neg',2)]:
                vals=rows[n].get('actual_mean');nums.append(f'{vals[a]:+.3f}' if vals else '无有效样本')
            survival=sum(rows[n]['survived'] for n in ('forward','backward','turn_pos','turn_neg'))
            score=f"{r['score']:.3f}" if r['score'] is not None else '无效'
            lines.append('| '+ ' | '.join([labels[r['variant']],*nums,str(survival),str(rows['stand']['survived']),score,
                                          '是' if r['all_basic_tasks_passed'] else '否'])+' |')
        lines+=['','逐任务指标和原始报告：']
        for r in results:
            path=Path(r['report_dir']).relative_to(root)/'report.json'
            lines.append(f"- [{labels[r['variant']]}]({path})")
        lines+=['']
    lines+=['## 判断范围','',
            '起步引导组同时改变负重转移、离地/交替落脚奖励和运动时姿态约束，是组合干预。即使有改善，也不能归因于其中一个单项。',
            '本实验检验停滞策略的短期恢复能力；1024 环境、300 轮结果不能代替 4096 环境的长期从零对照。抬脚更多、存活更多或训练奖励更高不等于走路通过。',
            '全通过只指本轮五项基础任务、低幅度手臂条件，不代表全幅 AMASS、推扰或深蹲组合通过。原正式训练及模型未被实验覆盖。','']
    if s['status']=='completed':
        lines.append('第二种子复核候选：'+str(s.get('replication_candidate'))+'。')
        if s.get('replication_reason'):lines.append(s['replication_reason'])
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(root/'report.md')


if __name__=='__main__':main()
