Why We Write This
arr[1, 2] → Direct element access. Faster than any loop. In image processing image[y, x] gives you the pixel at that coordinate.

arr[:2] → Slice rows. When you have a dataset of 1 million rows and want only the first 1000 for testing, this is how you do it — data[:1000].

arr[:, 2:] → The : means "all rows", 2: means "column 2 onwards". This pattern is everywhere — selecting specific features from a dataset without touching others.

arr.copy() → Critical habit. NumPy slices are views, not copies. If you modify a slice, you modify the original too. .copy() gives you an independent array. Forgetting this causes silent data corruption bugs in production pipelines.

arr[arr > 60] → Boolean indexing. This is pure NumPy magic. You pass a condition and get back only matching elements — no loops needed. Used in data cleaning to filter outliers, invalid values, or specific ranges.

arr[[0, 2]] → Fancy indexing — pick specific rows by index list. Used when you have a list of sample IDs and want to grab exactly those rows from a dataset.

Real Production Impact
In a data cleaning pipeline at a fintech company:

# Remove all transactions above $10,000 (potential fraud flag for review)
clean_data = transactions[transactions[:, 2] < 10000]

# Grab only specific customer records by ID index
selected = dataset[[customer_id_1, customer_id_2, customer_id_3]]
Both of these use exactly what you just learned.

Run it, share your output, then we move to Lesson 5: Reshaping and Stacking — where arrays change their form without losing their data. This is fundamental for feeding data into ML models.

Checkpoint
Restore
ok now move to another lesson

Kiro
Lesson 5: Reshaping and Stacking — Mold Your Data Into Any Shape
This is where NumPy starts feeling like a superpower. ML models are picky about input shapes — reshaping is how you feed them exactly what they expect.

Create lesson5.py and write this:
import numpy as np

# --- RESHAPING ---
arr = np.arange(1, 13)  # [1, 2, 3, ..., 12]
print("Original:", arr)
print("Shape:", arr.shape)

# Reshape to 2D — 3 rows, 4 columns
reshaped = arr.reshape(3, 4)
print("\nReshaped to (3,4):\n", reshaped)

# Reshape to 3D — 2 blocks, 2 rows, 3 columns
reshaped_3d = arr.reshape(2, 2, 3)
print("\nReshaped to (2,2,3):\n", reshaped_3d)

# Flatten back to 1D — reverse of reshape
flattened = reshaped.flatten()
print("\nFlattened:", flattened)

# -1 means numpy calculates the dimension for you
auto_reshape = arr.reshape(4, -1)  # 4 rows, numpy figures out columns
print("\nAuto reshape (4, -1):\n", auto_reshape)


# --- STACKING ---
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Stack vertically — adds rows
vstack = np.vstack([a, b])
print("\nVertical Stack:\n", vstack)

# Stack horizontally — adds columns
hstack = np.hstack([a, b])
print("\nHorizontal Stack:", hstack)

# Stack along new axis
stack = np.stack([a, b], axis=0)
print("\nStack axis=0:\n", stack)


# --- SPLITTING ---
arr2d = np.arange(1, 25).reshape(4, 6)
print("\nArray to split:\n", arr2d)

# Split into 3 equal parts along columns
splits = np.hsplit(arr2d, 3)
print("\nAfter hsplit into 3:")
for i, s in enumerate(splits):
    print(f"Part {i+1}:\n", s)
Why We Write This
reshape(3, 4) → Changes the shape without changing the data. The total elements must stay the same — 12 elements can be (3,4), (2,6), (1,12), (2,2,3) but never (3,5). This is the most used NumPy operation in ML.

reshape(2, 2, 3) → 3D arrays are how you represent a batch of matrices. A batch of 2 grayscale images each 2x3 pixels would be exactly this shape.

flatten() → Converts any multi-dimensional array to 1D. Before passing data into a fully connected neural network layer, you flatten the array. Every CNN ends with a flatten before the final dense layers.

reshape(4, -1) → The -1 is NumPy saying "you figure out one dimension, I'll calculate the other". Used constantly when you know the batch size but not the feature count, or vice versa.

np.vstack → Combines datasets by adding rows. Used when merging monthly data files into one annual dataset — stack January, February, March... vertically.

np.hstack → Combines by adding columns. Used when you have features computed separately and want to merge them into one feature matrix.

np.hsplit → Splits a dataset into parts. Used in cross-validation — split your data into k equal folds for k-fold validation.

Real Production Impact
In an image classification pipeline:

# Images come in as (28, 28) pixels each
# Batch of 64 images stacked together
batch = np.stack(list_of_64_images, axis=0)  # shape: (64, 28, 28)

# Flatten for feeding into dense layer
batch_flat = batch.reshape(64, -1)           # shape: (64, 784)
This exact pattern runs millions of times per day in production ML systems.

Run it, share your output, then we go to Lesson 6: Math Operations and Broadcasting — the lesson where NumPy's true speed advantage becomes obvious and you'll never want to write a Python loop for math again.
<!-- second changes made to learn git advance commands -->