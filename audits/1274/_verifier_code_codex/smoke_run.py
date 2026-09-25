"""Smoke test: run a tiny synthetic AMRIV experiment to confirm models.py runs and phi is finite. Supports finding repo-runs."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code", "CausalML__Adaptive-IV"))
os.chdir(os.path.join(os.path.dirname(__file__), "..", "code", "CausalML__Adaptive-IV"))
import numpy as np
from data import make_synthetic_iv_dgp, OracleMuA, OracleMuY, OracleSigma, true_ate, true_ate_var
from models import AMRIVExperiment, A2IPWExperiment
from utils import make_rf_factory

d=5
np.random.seed(1); beta=np.random.uniform(-1,1,size=d)
def my_f(X,a):
    X=np.asarray(X); single=X.ndim==1
    if single: X=X[None,:]
    val=1+a+X[:,0]+2*a*(X@beta)+0.75*a*X[:,0]**2
    return val[0] if single else val

gen=make_synthetic_iv_dgp(f=my_f,d=d,seed=0)
factories={
 "muY0": make_rf_factory(regression=True,n_estimators=20,max_depth=5,min_samples_leaf=5),
 "muY1": make_rf_factory(regression=True,n_estimators=20,max_depth=5,min_samples_leaf=5),
 "muA0": lambda **_: OracleMuA(z=0),
 "muA1": make_rf_factory(regression=False,n_estimators=20,max_depth=3,min_samples_leaf=30),
 "s0": make_rf_factory(regression=True,n_estimators=20,max_depth=5,min_samples_leaf=5),
 "s1": make_rf_factory(regression=True,n_estimators=20,max_depth=5,min_samples_leaf=5),
}
exp=AMRIVExperiment(generator=gen,n_rounds=601,burn_in=200,adaptive=True,batch_size=200,
                    deltaA_eps=1e-2,trunc_schedule=lambda t:2/(0.999)**t,factories=factories)
exp.collect()
phi=exp.phi[exp.burn_in+1:]
tt=true_ate(my_f,d,n=50000)
print("ran OK; n_phi=",len(phi),"mean_phi=",float(np.mean(phi)),"true_tau=",float(tt),"finite=",np.all(np.isfinite(phi)))
with open(os.path.join(os.path.dirname(__file__),"out","smoke.txt"),"w") as f:
    f.write(f"n_phi={len(phi)} mean_phi={float(np.mean(phi))} true_tau={float(tt)} finite={bool(np.all(np.isfinite(phi)))}\n")
