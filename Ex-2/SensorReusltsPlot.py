import matplotlib.pyplot as plt
#alle mål er transformeret til cm

#Resultater uden sleep
x = [10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 300, 300, 300, 300, 300]
y = [12.3, 12.5, 12.3, 12.3, 12.3, 50.1, 50.1, 50.1, 50.1, 50.1, 99.0, 99.0, 99.0, 98.9, 98.9, 198.8, 198.8, 198.8, 198.8, 198.8, 296.0, 295.6, 295.7, 295.6, 295.6]

x_10 = [10, 10, 10, 10, 10]
y_10 = [12.3, 12.5, 12.3, 12.3, 12.3]
# gennemsnitlig distance fra mål i cm: 2.34
# Variationsbredde: 0.2

x_50 = [50, 50, 50, 50, 50]
y_50 = [50.1, 50.1, 50.1, 50.1, 50.1]
# gennemsnitlig distance fra mål i cm: 0.1
# Variationsbredde: 0

x_100 = [100, 100, 100, 100, 100]
y_100 = [99.0, 99.0, 99.0, 98.9, 98.9]
# gennemsnitlig distance fra mål i cm: 1.04
# Variationsbredde: 0.1

x_200 = [200, 200, 200, 200, 200]
y_200 = [198.8, 198.8, 198.8, 198.8, 198.8]
# gennemsnitlig distance fra mål i cm: 1.2
# Variationsbredde: 0

x_300 = [300, 300, 300, 300, 300]
y_300 = [296.0, 295.6, 295.7, 295.6, 295.6]
# gennemsnitlig distance fra mål i cm: 4.3
# Variationsbredde: 0.4

#plt.scatter(x,y)
#plt.show()


#Resultater med sleep
xs = [10, 10, 10, 10, 10, 50, 50, 50, 50, 50, 100, 100, 100, 100, 100, 200, 200, 200, 200, 200, 300, 300, 300, 300, 300]
ys = [12.2, 11.8, 11.7, 11.7, 11.8, 51.4, 51.4, 51.4, 51.4, 51.4, 100.3, 100.8, 100.3, 100.3, 100.3, 198.6, 198.7, 198.7, 198.7, 198.6, 298.0, 298.0, 298.0, 298.0, 297.6]

xs_10 = [10, 10, 10, 10, 10]
ys_10 = [12.2, 11.8, 11.7, 11.7, 11.8]
# gennemsnitlig distance fra mål i cm: 1.84
# Variationsbredde: 0.5

xs_50 = [50, 50, 50, 50, 50]
ys_50 = [51.4, 51.4, 51.4, 51.4, 51.4]
# gennemsnitlig distance fra mål i cm: 1.4
# Variationsbredde: 0

xs_100 = [100, 100, 100, 100, 100]
ys_100 = [100.3, 100.8, 100.3, 100.3, 100.3]
# gennemsnitlig distance fra mål i cm: 0.4
# Variationsbredde: 0.5

xs_200 = [200, 200, 200, 200, 200]
ys_200 = [198.6, 198.7, 198.7, 198.7, 198.6]
# gennemsnitlig distance fra mål i cm: 1.34
# Variationsbredde: 0.1

xs_300 = [300, 300, 300, 300, 300]
ys_300 = [298.0, 298.0, 298.0, 298.0, 297.6]
# gennemsnitlig distance fra mål i cm: 2.08
# Variationsbredde: 0.4

plt.scatter(xs_10,ys_10)
plt.show()