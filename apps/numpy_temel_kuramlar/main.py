#numpy temel kullanımları

import numpy as np
"""
liste1 = np.array([1,2,3,4])
liste2 = np.array([5,6,7,8])

print(liste1*liste2)

print(np.zeros(12))

print(np.ones((4,6),dtype = int))

print(np.full((2,3), 10))

print(np.arange(0,10,2 ))

print(np.linspace(3,10,5))

print(np.random.normal(10,7,(2,5)))

print(np.random.randint(2,7,(2,5)))

#reshape kullanımı

liste1 = np.arange(1,11)

print(liste1)

liste2 = liste1.reshape(2, 5)

print(liste2)

liste3 = liste1.reshape(1,10)

print(liste3)


#concatenate kullanımı

liste1 = np.array([1,2,3,4])
liste2 = np.array([5,6,7,8])

liste3 = np.concatenate([liste1, liste2])

print(liste3)

liste4 = np.concatenate([[liste1,liste2]])

print(liste4)



#split , vsplit , hsplit

arr1 = np.arange(10)

print(arr1)

arr2 = np.split(arr1,[2,4])

print(arr2)

arr3,arr4,arr5 = np.split(arr1,[3,7])

print(arr4)

print(arr5)

print(arr3)

arr6 = np.arange(25).reshape(5,5)

arr7,arr8 = np.vsplit(arr6, [3])

print(arr7)

print(arr8)

arr9,arr10 = np.hsplit(arr6, [3])

print(arr9)

print(arr10)


#sort 

arr1 = np.array([1,5,7,2,3,4,91])

print(np.sort(arr1))

arr2 = np.random.normal(1,11,(4,4))

print(arr2)

print(np.sort(arr2, axis = 1))

print(np.sort(arr2, axis = 0))




#arraylerde index işlemleri

arr1 = np.array([10,2,3,4,5])
arr2 = np.arange(10)

print(arr1[2])
print(arr2[5])

arr3 = np.concatenate((arr1,arr2))

arr4 = arr3.reshape(3,5)

print(arr4[2][4])

arr4[2][4] = 10

print(arr4[2][4])

print(arr2[3:8])

print(arr4[:,2])

print(arr4[2])

print(arr4[1:3,2:3])

arr5 = arr4[1:3,2:3].copy()

istenilen_index_listesi = [1,5,7]

print(arr3[istenilen_index_listesi])



#arraylerde koşullu işlemler

arr1= np.arange(1,11)
arr2 = np.random.randint(1,11,(10))

print(arr1[arr1>5])
print(arr2[arr2<= 5])


"""
#2 bilinmeyenli denklem çözümü
# x1 + x2 = 15
# 2x1 + 3x2 = 47

denklem = np.array([[1,1],[2,3]])
deger = np.array([15,47])

sonuc = np.linalg.solve(denklem,deger)

print(sonuc)


