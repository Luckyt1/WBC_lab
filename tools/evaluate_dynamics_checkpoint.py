"""Frozen control/gait and auxiliary diagnostics for one dynamics checkpoint."""
import argparse
import os
from pathlib import Path
from legged_lab.paths import deployment_framework, deployment_mod, mujoco_xml
import subprocess
import sys

from run_focused_validation import ROOT, now, read, write
from run_scratch_training import health
from analyze_saved_walk_gait import analyze
from walk_robust_gates import evaluate_gates
from evaluation_runtime import deployment_python


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--deployment_python',type=Path,required=True)
    parser.add_argument('--reuse-isaac',action='store_true',
                        help='Validate and reuse completed Isaac traces; rerun deployment/probe stages only.')
    args=parser.parse_args()
    checked=health(args.checkpoint,std_bounds=(.1,.6))
    out=args.output
    out.mkdir(parents=True,exist_ok=True)
    args.deployment_python = deployment_python(args.deployment_python, out)
    write(out/'evaluation_runtime_used.json', dict(deployment_python=str(args.deployment_python),
          reuse_isaac=args.reuse_isaac, started_at=now()))
    subprocess.run([sys.executable,'-u','tools/evaluate_standing_candidate.py',
                    '--task','walk_elf3_tang_bxi_robust','--checkpoint',str(args.checkpoint),
                    '--output',str(out),'--deployment_python',str(args.deployment_python),
                    '--include-mixed-yaw','--gait-diagnostics','--auxiliary-probe']
                   + (['--reuse-isaac'] if args.reuse_isaac else []),cwd=ROOT,check=True)
    report=read(out/'report.json')
    assert report['checkpoint_sha256']==checked['sha256']
    gates=evaluate_gates(report)
    standing=read(out/'mujoco_standing.json')
    gait=analyze(out)
    write(out/'isaac_gait.json',gait)
    framework=deployment_framework()
    environment=dict(os.environ,PYTHONPATH=str(framework)+os.pathsep+os.environ.get('PYTHONPATH',''))
    subprocess.run([str(args.deployment_python),'-u','tools/diagnose_walk_mujoco.py',
                    '--mod',str(deployment_mod()),
                    '--xml',str(mujoco_xml()),
                    '--onnx',str(out/'policy.onnx'),'--output',str(out/'mujoco_gait'),
                    '--include-mixed-yaw'],cwd=ROOT,env=environment,check=True)
    subprocess.run([str(args.deployment_python),'-u','tools/evaluate_actor_auxiliary.py',
                    '--checkpoint',str(args.checkpoint),'--probes',str(out),
                    '--output',str(out/'auxiliary_prediction.json')],cwd=ROOT,check=True)
    mj=read(out/'mujoco_gait/report.json')['results']
    mixed=evaluate_gates(report,('arc_forward_pos','arc_forward_neg','arc_backward_pos','arc_backward_neg'),(.08,.08,.05))
    write(out/'summary.json',dict(checkpoint=str(args.checkpoint.resolve()),health=checked,
          completed_at=now(),gates=gates,original_gates_passed=gates['passed'] and standing['passed'],
          mixed_yaw_gates=mixed,standing=standing,isaac_forward=gait['profiles']['bootstrap']['scenarios']['forward_030'],
          mujoco_forward=mj['forward'],mujoco_fast=mj['fast'],mujoco_zero=mj['zero'],mujoco_walk_stop=mj['walk_stop'],
          report_dir=str(out),auxiliary_prediction_report=str(out/'auxiliary_prediction.json'),
          probe_method='Held-out rollouts of this checkpoint. Distribution changes with checkpoint; not a fixed cross-age dataset.',
          auto_promote=False))


def write_comparison(out,state):
    def fmt(value,digits=3):
        return '—' if value is None else f'{value:.{digits}f}'
    def duration(case):
        value=fmt(case.get('elapsed_s'),2)
        return value+(' 存活' if case['completed'] else ' 跌倒')
    target = state['target_total_iterations']
    continuation = not state['from_scratch']
    title = f"Dynamics 续训至累计 {target:,} 轮" if continuation else f"Dynamics 从零训练至 {target:,} 轮"
    method = (f"从 {state['resumed_checkpoint']} 恢复策略、辅助预测分支及两个优化器，保留原训练课程。"
              if continuation else '只训练 dynamics；原 control/state 文件保留。初始策略随机、优化器为空，没有继承旧训练权重。')
    if state.get('pelvis_height') is not None:
        method += f" 本段目标骨盆高度为 {state['pelvis_height']} m；评估采用相同高度指令。"
    if state.get('teleop_curriculum_stage'):
        method += f" 本段课程固定为 {state['teleop_curriculum_stage']}；外部课程控制器在本段结束后串行评估，并决定留级、升阶或回退难度。"
    lines=[f"# {title}",'',
           f"状态：`{state['status']}`；最近更新：{state['updated_at']}。",
           f"本段已记录更新：{state['completed_updates']:,} / {state['planned_updates']:,}；累计 {state['initial_iteration'] + state['completed_updates']:,} / {target:,}。",'',
           method,
           '测试时长仅表示存活情况；是否按指令行走必须同时看实际速度和速度误差。','',
           '| 检查点 | 直行测试时长 | 实际前向速度（目标 0.3 m/s） | 前向速度 MAE | 高速实际速度（目标 0.6 m/s） | 静止测试 | 静止最大漂移 cm | 完整摆动 L/R |',
           '|---|---|---|---|---|---|---|---|']
    for row in state['evaluations']:
        forward,fast,zero=row['mujoco_forward'],row['mujoco_fast'],row['mujoco_zero']
        def vx(case,key):
            return case.get(key,[None])[0]
        drift=zero.get('max_planar_offset_m')
        counts='/'.join(str(x) for x in row['isaac_forward'].get('supported_swing_120_350ms_20mm_count',[]))
        lines.append(f"| {row['health']['iteration']} | {duration(forward)} | {fmt(vx(forward,'mean_velocity_yaw'))} | "
                     f"{fmt(vx(forward,'velocity_mae'))} | {fmt(vx(fast,'mean_velocity_yaw'))} | {duration(zero)} | "
                     f"{fmt(drift*100 if drift is not None else None,2)} | {counts} |")
    lines += ['', '完整摆动仍采用 120–350 ms 腾空、足底至少 20 mm、另一脚持续支撑的诊断定义。',
              '每个检查点另记录三类扰动、移动转向、航向变化和走停漂移；详细结果在 evaluations/ 对应目录。',
              '辅助预测使用该检查点自己的独立测试轨迹；不同年龄的数据分布不同，误差下降不单独证明控制改善。',
              '单种子长程训练不构成跨种子验证。任何模型都不自动部署。','']
    for error in state.get('evaluation_errors',[]):
        lines.append(f"评估程序异常：{error['checkpoint']}，退出码 {error['returncode']}。")
    (out/'RESULTS.md').write_text('\n'.join(lines))


if __name__=='__main__':
    main()
