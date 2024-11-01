import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. Rastgele bir tohum ayarlayarak sonuçların tekrarlanabilir olmasını sağlıyoruz
np.random.seed(42)

# 2. Yapay veri seti oluşturma
class_0 = np.random.multivariate_normal(mean=[2, 2],
                                       cov=[[1, 0.5], [0.5, 1]],
                                       size=100)
class_1 = np.random.multivariate_normal(mean=[4, 4],
                                       cov=[[1, -0.5], [-0.5, 1]],
                                       size=100)
data = np.vstack((class_0, class_1))
labels = np.hstack((np.zeros(100), np.ones(100)))

# 3. Veri setini DataFrame'e dönüştürme
df = pd.DataFrame(data, columns=['x1', 'x2'])
df['label'] = labels
print(df.head())

# 4. Veri görselleştirme
plt.figure(figsize=(8,6))
sns.scatterplot(x='x1', y='x2', hue='label', data=df, palette='viridis')
plt.title('Yapay Veri Seti Dağılımı')
plt.show()

# 5. Veri setini eğitim ve test olarak bölme
X = df[['x1', 'x2']]
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# 6. Lojistik regresyon modelini oluşturma ve eğitme
model = LogisticRegression()
model.fit(X_train, y_train)

# 7. Modelin değerlendirilmesi
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Doğruluk Oranı: {accuracy:.2f}")

print("\nSınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Tahmin Edilen Etiket')
plt.ylabel('Gerçek Etiket')
plt.title('Confusion Matrix')
plt.show()

# 8. Karar sınırını görselleştirme
h = 0.02
x_min, x_max = X['x1'].min() - 1, X['x1'].max() + 1
y_min, y_max = X['x2'].min() - 1, X['x2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(8,6))
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
sns.scatterplot(x='x1', y='x2', hue='label', data=df, palette='viridis', edgecolor='k')
plt.title('Lojistik Regresyon Karar Sınırı')
plt.show()
