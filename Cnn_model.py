import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns



train_dir = "Fruits_Vegetables/Fruits_Vegetables/train"
validation_dir = "Fruits_Vegetables/Fruits_Vegetables/validation"
test_dir = "Fruits_Vegetables/Fruits_Vegetables/test"


IMG_SIZE = (128, 128)
BATCH_SIZE = 32


train_data = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_data = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_data.class_names

print("Classes:", class_names)


model = tf.keras.models.load_model(
    "fruit_vegetable_model.keras"
)


test_loss, test_accuracy = model.evaluate(test_data)

print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)


plt.figure(figsize=(10, 10))

for images, labels in test_data.take(1):

    predictions = model.predict(images, verbose=0)

    for i in range(9):

        predicted_label = class_names[
            np.argmax(predictions[i])
        ]

        actual_label = class_names[
            labels[i]
        ]

        plt.subplot(3, 3, i + 1)

        plt.imshow(
            images[i].numpy().astype("uint8")
        )

        plt.title(
            f"Actual: {actual_label}\n"
            f"Predicted: {predicted_label}"
        )

        plt.axis("off")

plt.show()


y_true = []
y_pred = []

for images, labels in test_data:

    predictions = model.predict(
        images,
        verbose=0
    )

    y_true.extend(labels.numpy())

    y_pred.extend(
        np.argmax(predictions, axis=1)
    )

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(figsize=(12, 10))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

from sklearn.metrics import classification_report

print(classification_report(y_true, y_pred, target_names=class_names))

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

scores = {
    class_name: report[class_name]["f1-score"]
    for class_name in class_names
}

best_class = max(scores, key=scores.get)
worst_class = min(scores, key=scores.get)

print("Best Class:", best_class)
print("Best F1 Score:", round(scores[best_class], 2))

print("Weakest Class:", worst_class)
print("Weakest F1 Score:", round(scores[worst_class], 2))