from pathlib import Path
import math,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from m11_models import cubic_coefficients,advance_cubic,wrap_angle

for d in [15.8-1e-7,15.8,15.8+1e-7,25.]:
    x,a1,a2,a3,rotation,valid=cubic_coefficients(0.,0.,0.,d*math.cos(.2),d*math.sin(.2),.3)
    assert valid
    target=wrap_angle(rotation+math.atan(a1+2*a2*x+3*a3*x*x))
    assert abs(target-(.2 if d<=15.8 else .3))<1e-12
    lx,ly,heading=advance_cubic(a1,a2,a3,x,0.)
    assert abs(wrap_angle(rotation+heading))<1e-12
    endpoint=a1*x+a2*x*x+a3*x*x*x
    global_x=math.cos(rotation)*x-math.sin(rotation)*endpoint
    global_y=math.sin(rotation)*x+math.cos(rotation)*endpoint
    assert abs(global_x-d*math.cos(.2))<1e-10
    assert abs(global_y-d*math.sin(.2))<1e-10
assert not cubic_coefficients(0.,0.,0.,0.,0.,0.)[-1]
assert cubic_coefficients(0.,0.,0.,10.,2.,math.pi-.004)[-1]
assert not cubic_coefficients(0.,0.,0.,30.,6.,math.pi-.004)[-1]
assert cubic_coefficients(0.,0.,math.pi-.05,-20.,-2.,-math.pi+.05)[-1]
print('PASS: threshold sides, exact threshold, endpoint constraints, zero advance, wrap boundary and excluded domains')
