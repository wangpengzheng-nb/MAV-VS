#!/usr/bin/env python3
"""AF3 API 连通性测试"""
import os, json, time
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from autovs.af3 import af3_env_available, load_af3_env, af3_health, _submit_job

# 1. 认证检查
ok, reason = af3_env_available()
print(f'1. Auth check: {"✅" if ok else "❌"} {reason}')

# 2. 健康检查
ok2, msg = af3_health()
print(f'2. Health: {"✅" if ok2 else "❌"} {msg[:150]}')

if ok2:
    env = load_af3_env()
    print(f'   Server: {env.server_url}')

    # 3. 提交BCL-2测试任务
    bcl2_seq = ('MAHAGRTGYDNREIVMKYIHYKLSQRGYEWDAGDVGAAPPGAAPAPGIFSSQPGHTPHPA'
                'ASRDPVARTSPLQTPAAPGAAAGPALSPVPPVVHLTLRQAGDDFSRRYRRDFAEMSSQLH'
                'LTPFTARGRFATVVEELFRDGVNWGRIVAFFEFGGVMCVESVNREMSPLVDNIALWMTEY'
                'LNRHLHTWIQDNGGWDAFVELYGPSMRPLFDFSWLSLKTLLSLALVGACITLGAYLGHK')

    payload = {
        'name': 'bcl2_test',
        'dialect': 'alphafold3',
        'version': 1,
        'modelSeeds': [1],
        'sequences': [{'protein': {'id': 'A', 'sequence': bcl2_seq}}]
    }

    print(f'\n3. 提交BCL-2预测任务...')
    print(f'   序列长度: {len(bcl2_seq)} aa')
    t0 = time.time()
    try:
        result = _submit_job(env, payload, name='bcl2_test')
        job_id = result.get('job_id', result.get('id', '?'))
        status = result.get('status', '?')
        print(f'   ✅ 提交成功: {job_id} status={status} ({time.time()-t0:.1f}s)')

        # 4. 轮询状态
        from autovs.af3 import _request
        print(f'\n4. 轮询作业状态...')
        for i in range(20):
            time.sleep(15)
            code, body = _request(env, "GET", f"/api/jobs/{job_id}", timeout=30)
            data = json.loads(body)
            state = data.get('status', '?')
            print(f'   [{i+1}] {state}', end=' ', flush=True)
            if state in ('succeeded', 'failed', 'cancelled'):
                print(f'\n   最终状态: {state}')
                if state == 'succeeded':
                    print(f'   ✅ AF3预测成功!')
                break
            print('...')
    except Exception as e:
        print(f'   ❌ 失败: {e}')
else:
    print('\n❌ AF3服务不可用，无法提交任务')
