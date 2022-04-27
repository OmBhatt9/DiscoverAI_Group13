from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

digits = datasets.load_digits()
print(digits.data)
print(digits.target)

plt.gray() 

plt.matshow(digits.images[10])

plt.show()
print(digits.target[10])

#Figure size (width, height)
fig = plt.figure(figsize=(6, 6))

#Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

#For each of the 64 images
for i in range(64):

    #Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])

    #Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')

    #Label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

plt.show()

# Elbow method to choose K

sse_1 = []
k_rng1 = range(1,80)
for k in k_rng1:
    km = KMeans(n_clusters=k)
    km.fit(digits.data)
    sse_1.append(km.inertia_)

plt.figure()
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng1,sse_1)
plt.xticks(range(1, 80, 3))


n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.2, random_state = 45)

# clusters = 10 as we saw from above

model = KMeans(n_clusters=10, random_state = 45)
model.fit(X_train, y_train)


plt.figure(figsize=(8, 3))
plt.suptitle('Cluster Center Images', fontsize=14, fontweight='bold')

for i in range(10):
  ax = fig.add_subplot(2, 5, 1 + i, xticks=[], yticks=[])

  #Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)),  cmap=plt.cm.binary)
plt.show()


#Predicting numbers
predicted = model.fit_predict(X_test)

###############################################################################
# visualize the first 8 test samples and show their predicted
# digit value in the title.

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, prediction, actual in zip(axes, X_test, predicted, y_test):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {prediction} actual: {actual}")


print(
    f"Classification report for classifier {model}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n"
)