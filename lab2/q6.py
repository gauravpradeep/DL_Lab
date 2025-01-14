import torch

x = torch.tensor(2.,requires_grad=True)
y = torch.tensor(1.,requires_grad=True)
z = torch.tensor(3.,requires_grad=True)

a=(2*x)
a.retain_grad()
b=(torch.sin(y))
b.retain_grad()
c=a/b
c.retain_grad()
d=c*z
d.retain_grad()
e=torch.log(d+1)
e.retain_grad()
f=torch.tanh(e)

f.backward()

print("AUTOGRAD")
print(f"x : {x} x.grad: {x.grad}")
print(f"y : {y} y.grad: {y.grad}")
print(f"z : {z} z.grad: {z.grad}")
print(f"a : {a} a.grad: {a.grad}")
print(f"b : {b} b.grad: {b.grad}")
print(f"c : {c} c.grad: {c.grad}")
print(f"d : {d} d.grad: {d.grad}")
print(f"e : {e} e.grad: {e.grad}")

print("Analytical")
def analytical_gradient():
    df_de = 1 - f**2
    de_dd = 1 / (d + 1)
    dd_dc = z
    dc_db = -a / b**2
    db_dy = torch.cos(y)
    print(f"df_dy: {df_de * de_dd * dd_dc * dc_db * db_dy}")

analytical_gradient()

