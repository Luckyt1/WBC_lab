"""Write a reviewable, continuously updated comparison for bounded gait runs."""
import argparse
import json
from pathlib import Path


def summarize_batch(out, state):
    updates = state.get('updates_per_variant', 500)
    num_envs = state.get('num_envs', 4096)
    seed = state.get('seed', 42)
    fresh = state.get('from_scratch', False)
    source_iteration = (state.get('source_health') or {}).get('iteration', 26000)
    rows = []
    entries = [(f'reference_{i}', summary) for i, summary in enumerate(state.get('references', []))]
    entries += [(variant['variant'], summary) for variant in state['variants'] for summary in variant['evaluations']]
    for label, summary in entries:
        gait = json.loads(Path(summary['gait_report']).read_text())
        mj_report = json.loads(Path(summary['deployment_diagnostics']).read_text())
        mj = mj_report['results']
        forward = gait['profiles']['bootstrap']['scenarios']['forward_030']
        standing = mj['zero']
        arcs = {k: dict(completed=v['completed'], velocity_mae=v.get('velocity_mae'))
                for k, v in mj.items() if k.startswith('arc_')}
        row = dict(variant=label, iteration=summary['health']['iteration'], checkpoint=summary['checkpoint'],
                   original_gates_passed=summary['original_gates_passed'],
                   mixed_gates_passed=summary.get('mixed_yaw_gates', {}).get('passed'),
                   mujoco_terrain=mj_report.get('terrain', {'mode':'legacy/unrecorded'}),
                   isaac_forward=forward, mujoco_forward=mj['forward'], mujoco_fast=mj['fast'],
                   mujoco_zero=standing, mujoco_walk_stop=mj['walk_stop'], mujoco_arcs=arcs)
        if 'auxiliary_prediction_report' in summary:
            row['auxiliary_prediction_report']=summary['auxiliary_prediction_report']
        rows.append(row)
    result = dict(status=state['status'], source_checkpoint=state['source_checkpoint'],
                  updated_at=state['updated_at'], auto_promote=False, rows=rows,
                  method=f'Same extended benchmark for source, matched {updates}-update control and new variants; '
                         'Isaac has 32 trials per command/profile. MuJoCo uses one deterministic rollout per case. '
                         'Raw contacts can chatter. Supported-swing counts also require 120–350 ms air time and 20 mm sole clearance.',
                  limitation='A single training seed cannot establish robustness across training seeds.')
    (out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    def fmt(value, places=2):
        return '—' if value is None else f'{value:.{places}f}'
    auxiliary=state.get('study')=='actor_auxiliary'
    title='# 辅助状态监督验证' if auxiliary else '# 第二轮步态与漂移验证'
    reference=('reference_0 是原始候选；control 是同预算原配方，state 增加当前速度/接触监督，dynamics 另加动作条件下一状态预测。'
               if auxiliary else 'reference_0 是原始候选；reference_1 是上一轮未改奖励的 500 次更新对照。所有模型重跑相同的扩展测试。')
    description = f'保留 {source_iteration:,} 轮候选。每个新分支独立续训 {updates:,} 次更新，使用相同初始权重、优化器、{num_envs} 环境和 seed {seed}。'
    if fresh:
        title = '# 从零训练：辅助状态监督验证'
        description = f'三组均从随机策略和空优化器开始，各训练 {updates:,} 次更新，使用相同初始网络、训练课程、{num_envs} 环境和 seed {seed}。'
        reference = 'control 是从零训练原配方；state 增加速度/接触监督；dynamics 再加动作条件下一状态预测。没有加载旧模型或教师。5,000 轮是首段学习比较，不代表已收敛。'
        result['method'] = result['method'].replace('for source, matched', 'for matched fresh-initialization')
        result['initializations'] = state.get('initializations', {})
        (out/'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    lines = [title, '', f"状态：`{state['status']}`；更新时间：{state['updated_at']}。", '',
             description,
             reference, '',
             '| 分支 / 轮数 | MuJoCo 地形 | 原有验收 | 移动转向验收 | Isaac 腾空中位数 L/R（ms） | 完整支撑摆动 L/R（次） | MuJoCo 直行时长（s） | 直行航向变化（°） | 高速时长（s） | 静止漂移（cm） |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        f=r['isaac_forward']; m=r['mujoco_forward']
        air='/'.join(fmt(f[s]['median_air_s']*1000 if f[s]['median_air_s'] is not None else None, 0) for s in ('left','right'))
        counts='/'.join(str(x) for x in f.get('supported_swing_120_350ms_20mm_count', []))
        elapsed=fmt(m.get('elapsed_s'))+(' ✓' if m['completed'] else ' 跌倒')
        drift=r['mujoco_zero'].get('max_planar_offset_m')
        terrain = r['mujoco_terrain']['mode']
        lines.append(f"| {r['variant']} / {r['iteration']} | {terrain} | {'通过' if r['original_gates_passed'] else '未通过'} | "
                     f"{'通过' if r['mixed_gates_passed'] else '未通过'} | {air} | {counts} | {elapsed} | "
                     f"{fmt(m.get('yaw_change_deg'))} | {fmt(r['mujoco_fast'].get('elapsed_s'))} | {fmt(drift*100 if drift is not None else None)} |")
    lines += ['', '地形 flat 为平地，source 保留 XML 障碍物；legacy/unrecorded 的旧结果未记录地形，不能直接比较跌倒率。', '完整支撑摆动：仅统计测量窗口内完整的 120–350 ms 腾空，脚底至少抬起 20 mm，且另一脚持续支撑。统计限于存活试次；它不是单独的部署通过条件。',
              '新增移动转向验收：前进/后退各 ±0.1 rad/s，速度 MAE 上限为 0.08、0.08、0.05；仍要求每组至少 31/32 存活及膝关节限幅通过。',
              'MuJoCo 直行目标持续 60 秒，高速目标持续 20 秒；航向变化仅应在相同测试时长、均未跌倒时比较。',
              '单次种子训练用于比较方向，不代表跨种子稳定性。流程不会自动替换部署模型。', '']
    if auxiliary:
        probe_description = ('辅助预测使用相同训练年龄的从零 control 生成的固定探针数据；仅用于评估，不参与训练。不同年龄的探针分布不同，不能直接把跨年龄误差下降当作预测进步。'
                             if fresh else '辅助预测在同一份原始 26k 策略产生的固定探针数据上比较，数据不参与训练。')
        lines += [probe_description+'关节/角速度预测另与“保持当前速度”的基线比较。','']
        for row in rows:
            if 'auxiliary_prediction_report' in row:
                path=Path(row['auxiliary_prediction_report'])
                lines.append(f"- {row['variant']} / {row['iteration']}：[辅助预测指标]({path.relative_to(out)})")
    (out/'RESULTS.md').write_text('\n'.join(lines))
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', type=Path, required=True)
    args=parser.parse_args()
    summarize_batch(args.batch, json.loads((args.batch/'state.json').read_text()))
