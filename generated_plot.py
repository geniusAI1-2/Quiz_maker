
import matplotlib.pyplot as plt
import numpy as np

# تحديد زوايا الدائرة من 0 إلى 2π (360 درجة)
theta = np.linspace(0, 2*np.pi, 100)

# حساب قيم x و y بناءً على المعادلة البارامترية للدائرة
r = 5
x = r * np.cos(theta)
y = r * np.sin(theta)

# إنشاء الرسم البياني
fig, ax = plt.subplots(1)

# رسم الدائرة
ax.plot(x, y)
ax.set_aspect(1) # لجعل الدائرة تبدو دائرية وليست بيضاوية

# إضافة عنوان
plt.title('Circle with radius 5 centered at the origin')

# إضافة تسميات للمحاور
plt.xlabel('x')
plt.ylabel('y')

# إضافة شبكة
plt.grid(True)

# حفظ الرسم البياني كصورة
plt.savefig('plot.png')
