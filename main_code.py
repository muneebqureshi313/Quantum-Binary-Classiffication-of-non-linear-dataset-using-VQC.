!pip install qiskit qiskit-machine-learning pennylane scikit-learn matplotlib
!pip install qiskit qiskit-aer qiskit-machine-learning pennylane scikit-learn matplotlib
!pip uninstall -y qiskit
!pip install qiskit==0.45.0 qiskit-aer==0.13.0 qiskit-machine-learning==0.6.1 pennylane scikit-learn matplotlib
from qiskit import QuantumCircuit
from qiskit_aer import Aer
from qiskit.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Generate synthetic 2D data
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Visualize the data
plt.figure(figsize=(6, 5))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.title("Training Data Distribution (make_moons)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()


from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

def create_hybrid_encoded_circuit(n_qubits=2, reps=3):
    """
    Create a variational quantum circuit with hybrid encoding:
    - Angle embedding of classical data re-uploaded in each layer.
    - Trainable RY + RZ rotations per layer.
    - Entanglement (CX) per layer.
    """
    # Parameters for input features (reuploaded in each layer)
    x = ParameterVector('x', length=n_qubits)

    # Parameters for trainable weights
    theta = ParameterVector('θ', length=reps * n_qubits * 2)

    qc = QuantumCircuit(n_qubits)
    param_count = 0

    for layer in range(reps):
        # Encode input data (data reuploading in each layer)
        for qubit in range(n_qubits):
            qc.ry(x[qubit], qubit)

        # Add trainable RY and RZ rotations
        for qubit in range(n_qubits):
            qc.ry(theta[param_count], qubit)
            param_count += 1
            qc.rz(theta[param_count], qubit)
            param_count += 1

        # Entanglement layer (linear entanglement)
        for qubit in range(n_qubits - 1):
            qc.cx(qubit, qubit + 1)

    return qc, x, theta


qc, x, theta = create_hybrid_encoded_circuit()
qc.draw("mpl")

!pip install torch
import torch
import torch.nn as nn
import torch.optim as optim
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.opflow import PauliSumOp
from qiskit_aer import Aer

# Get circuit and parameters
qc, x_params, θ_params = create_hybrid_encoded_circuit()

# Define observable (Z ⊗ I for binary classification)
observable = PauliSumOp.from_list([("Z" + "I", 1.0)])

qnn = CircuitQNN(
    circuit=qc,
    input_params=x_params,
    weight_params=θ_params,
    quantum_instance=Aer.get_backend('aer_simulator_statevector')
)

# Wrap QNN inside a PyTorch layer
model = TorchConnector(qnn)
# Convert to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)
# For binary classification with raw scores
loss_fn = nn.BCEWithLogitsLoss()

# Convert labels to float (for BCE)
y_train_float = y_train_tensor.float().unsqueeze(1)

import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, qnn):
        super().__init__()
        self.qnn = TorchConnector(qnn)
        self.linear = nn.Linear(4, 1)  # 4 outputs -> 1 output

    def forward(self, x):
        x = self.qnn(x)            # Output shape: [batch, 4]
        x = self.linear(x)         # Project to [batch, 1]
        return x                   # Final output (logit)
model = HybridModel(qnn)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
y_train_float = y_train_tensor.float().unsqueeze(1)  # make shape [N, 1]

epochs = 25
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = loss_fn(output, y_train_float)
    loss.backward()
    optimizer.step()

    # Calculate accuracy
    with torch.no_grad():
        preds = (torch.sigmoid(output) > 0.5).int().squeeze()
        acc = (preds == y_train_tensor).float().mean()

    print(f"Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Train Acc: {acc:.4f}")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convert test data
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test)

# Make predictions
with torch.no_grad():
    logits = model(X_test_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).int().squeeze()

# Metrics
acc = accuracy_score(y_test_tensor, preds)
prec = precision_score(y_test_tensor, preds)
rec = recall_score(y_test_tensor, preds)
f1 = f1_score(y_test_tensor, preds)

print(f"Test Accuracy:  {acc:.4f}")
print(f"Precision:      {prec:.4f}")
print(f"Recall:         {rec:.4f}")
print(f"F1 Score:       {f1:.4f}")

import numpy as np
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    h = 0.1
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_tensor = torch.tensor(grid, dtype=torch.float32)

    with torch.no_grad():
        logits = model(grid_tensor)
        Z = torch.sigmoid(logits).reshape(xx.shape)

    plt.contourf(xx, yy, Z, levels=[0, 0.5, 1], cmap='coolwarm', alpha=0.6)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    plt.title("Decision Boundary (Quantum Classifier)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True)
    plt.show()

# Plot using full dataset
plot_decision_boundary(model, X_scaled, y)

from sklearn.svm import SVC

clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)
print("Classical SVM Test Accuracy:", clf.score(X_test, y_test))














































