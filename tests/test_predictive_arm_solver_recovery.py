"""A failed position solve must produce a checked stop, never a frozen state."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_predictive_arm import fixture, native, guard


def inject_failure(executor, code=-110, corrupt_brake=None):
    calls=[]

    def calculate(state, goal, times, velocity, acceleration, jerk, brake, threads):
        calls.append((state.copy(), brake))
        out, bounds, status=native.calculate(state, goal, times, velocity, acceleration, jerk, brake, threads)
        if len(calls)==1:
            status[0]=code
            # The native error placeholder has unchanged p/v/a. It must never
            # be used as a real trajectory; poison it to catch accidental use.
            out[0]=np.nan;bounds[0]=np.nan
        elif brake and corrupt_brake:
            corrupt_brake(out, bounds, status)
        return out, bounds, status

    executor.native=SimpleNamespace(calculate=calculate)
    return calls


def test_only_failed_rows_brake_from_the_original_state():
    _,e,q,_=fixture()
    e.state[:,1]=.4;e.state[:,2]=2.
    before=e.state.copy();goal=q.numpy().astype('d')+.2
    expected=native.calculate(before,goal,e.times,e.vmax,e.amax,e.jmax,False,4)
    stopped=native.calculate(before[:1],goal[:1],e.times,e.vmax,e.amax,e.jmax,True,4)
    calls=inject_failure(e)
    with pytest.warns(RuntimeWarning,match='bounded braking'):
        out,bounds=e._calculate(before,goal)
    np.testing.assert_array_equal(out[0],stopped[0][0])
    np.testing.assert_array_equal(bounds[0],stopped[1][0])
    np.testing.assert_array_equal(out[1],expected[0][1])
    np.testing.assert_array_equal(bounds[1],expected[1][1])
    np.testing.assert_array_equal(calls[1][0],before[:1])
    np.testing.assert_array_equal(e.state,before)
    assert len(calls)==2 and calls[1][1] and e.solver_brake_total==1
    assert np.abs(out[0,0,0]-before[0,0]).max()>0
    assert np.abs(out[0,-1,1:]).max()<1e-12
    assert np.abs(out[0,:,1]).max()<=e.vmax
    assert np.abs(out[0,:,2]).max()<=e.amax
    assert np.abs(out[0,0,2]-before[0,2]).max()<=e.jmax*e.dt


@pytest.mark.parametrize('code,brake',[(-100,False),(-111,False),(-1000,False),(-110,True)])
def test_other_errors_and_failed_braking_still_stop_execution(code,brake):
    _,e,q,_=fixture();calls=inject_failure(e,code)
    with pytest.raises(RuntimeError,match='solver failed'):
        e._calculate(e.state,q.numpy(),brake=brake)
    assert len(calls)==1 and e.solver_brake_total==0


@pytest.mark.parametrize('field',['state','goal'])
def test_nonfinite_input_is_not_recovered(field):
    _,e,q,_=fixture();state=e.state.copy();goal=q.numpy().copy()
    (state if field=='state' else goal).flat[0]=np.nan
    calls=inject_failure(e)
    with pytest.raises(RuntimeError,match='non-finite state or goal'):
        e._calculate(state,goal)
    assert calls==[]


@pytest.mark.parametrize('failure',['status','velocity','acceleration','jerk','nan','bounds'])
def test_invalid_stop_is_never_accepted(failure):
    _,e,q,_=fixture()

    def corrupt(out,bounds,status):
        if failure=='status':status[:]=-110
        elif failure=='velocity':out[:,0,1,0]=e.vmax+1.
        elif failure=='acceleration':out[:,0,2,0]=e.amax+1.
        elif failure=='jerk':out[:,0,2,0]=e.jmax*e.dt+1.
        elif failure=='nan':out[:,0,0,0]=np.nan
        elif failure=='bounds':bounds[:]=np.nan

    inject_failure(e,corrupt_brake=corrupt)
    with pytest.raises(RuntimeError,match='brake fallback failed or exceeded limits'):
        e._calculate(e.state,q.numpy())
    assert e.solver_brake_total==0


def test_brake_fallback_goes_through_collision_prediction_and_telemetry():
    g,e,q,names=fixture()
    e.state[:,1]=.1
    pos=torch.zeros(2,len(names),3);rot=torch.zeros(2,len(names),4);rot[...,0]=1
    pos[:,1]=10.
    p,r=guard.arm_frames(q.reshape(2,2,7));pos[:,2:]=p.reshape(2,8,3);rot[:,2:]=r.reshape(2,8,4)
    goal=q.numpy().astype('d')
    stopped=native.calculate(e.state[:1],goal[:1],e.times,e.vmax,e.amax,e.jmax,True,4)[0]
    checked=[];original=e._clearance

    def clearance(preview,*args,**kwargs):
        checked.append(preview.detach().cpu().numpy().copy())
        return original(preview,*args,**kwargs)

    e._clearance=clearance;inject_failure(e)
    with pytest.warns(RuntimeWarning,match='bounded braking'):
        result=e.project(q,q,pos,rot,0)
    np.testing.assert_allclose(checked[0][:1],stopped,rtol=1e-6,atol=1e-7)
    assert torch.isfinite(result).all()
    assert e.metrics['ArmGuard/solver_brake_count']==1
    assert e.metrics['ArmGuard/solver_brake_total']==1
