import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def load_batch(file_path):
    with open(file_path, 'rb') as f:
        data_dict = pickle.load(f, encoding='bytes')
    return data_dict

# Veri yolu tanımı
path = r'C:\Users\Melih\Desktop\makine öğrenmesi\cifar-10-batches-py'

# Eğitim ve test verilerinin okunması
train_data = load_batch(os.path.join(path, 'data_batch_1'))
X_train = train_data[b'data']
y_train = np.array(train_data[b'labels'])

test_data = load_batch(os.path.join(path, 'test_batch'))
X_test = test_data[b'data'][:500] 
y_test = np.array(test_data[b'labels'][:500])

# Parametre testleri
k_values = [1, 3, 5, 7, 11]
accuracy_list = []

print(f"{'k':<5} | {'Dogruluk':<10}")
print("-" * 20)

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    accuracy_list.append(acc)
    
    print(f"{k:<5} | %{acc*100:.2f}")

# Grafik çizimi
plt.figure(figsize=(7, 4))
plt.plot(k_values, accuracy_list, 'b-o', label='Dogruluk Orani')
plt.title('CIFAR-10 KNN Sonuclari')
plt.xlabel('Komsu Sayisi (k)')
plt.ylabel('Skor')
plt.legend()
plt.grid(True)
plt.show()