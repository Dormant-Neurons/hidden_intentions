import argparse
from langchain_ollama import OllamaEmbeddings
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
from sklearn.metrics import roc_curve, auc


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x).squeeze()


def make_tensors(data, embeddings, label=None):
    X = []
    y = []

    for _, row in data.iterrows():
        if label is not None and row['Labels'] != label:
            continue

        X.append(f"Q: {row['Prompts']} \n A: {row['Responses']}")
        y.append(0 if row['Labels'] == 'Yes' else 1)

    X_embed = []
    for x in X:
        X_embed.append(torch.tensor(embeddings.embed_query(x)))

    return torch.stack(X_embed), torch.tensor(y, dtype=torch.float32)


def plot_tsne(X_tensor, y_tensor):
    X_array = X_tensor.numpy()
    y_array = y_tensor.numpy().astype(int)

    tsne = TSNE(n_components=2, random_state=0, perplexity=30)
    X_tsne = tsne.fit_transform(X_array)
    plot_df = pd.DataFrame({
        'x': X_tsne[:, 0],
        'y': X_tsne[:, 1],
        'label': y_array
    })

    plt.figure(figsize=(8, 6))
    ax = sns.scatterplot(data=plot_df, x='x', y='y', hue='label', palette='Set1', alpha=0.7)
    sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
    plt.title('t-SNE of QA Pair Embeddings')
    plt.show()


def evaluate_alternate(model, X_tensor, y_tensor, device):
    test_dataset = TensorDataset(X_tensor, y_tensor)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.sigmoid(model(xb)) > 0.5
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    print(f'Test Accuracy: {correct / total:.2f}')


def run_experiment(X_tensor, y_tensor, X_new_tensor, y_new_tensor, device):
    plot_tsne(X_tensor, y_tensor)

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    model = SimpleClassifier(input_dim=X_tensor.shape[1]).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(50):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f'Epoch {epoch}, Loss: {total_loss:.4f}')
        losses.append(total_loss)

        if epoch == 49:
            plt.figure(figsize=(10, 6))
            plt.plot(losses, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Over Time')
            plt.legend()
            plt.grid(True)
            plt.show()

    all_probs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    plt.figure(figsize=(8, 6))
    sns.histplot(x=all_probs, hue=all_labels, bins=20, kde=True, palette='Set1', stat='density')
    plt.title('Histogram of Predicted Probabilities')
    plt.xlabel('Predicted Probability (class 1)')
    plt.ylabel('Density')
    plt.legend(title='True Label', labels=['1 = Normal Response', '0 = Strategic Vagueness'])
    plt.show()

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = torch.sigmoid(model(xb)) > 0.5
            correct += (preds == yb).sum().item()
            total += yb.size(0)

    print(f'Test Accuracy: {correct / total:.2f}')

    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()

    evaluate_alternate(model, X_new_tensor, y_new_tensor, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', required=True)
    args = parser.parse_args()

    prompts = pd.read_csv('../Testbed/prompts.csv')
    responses = pd.read_csv('../Testbed/responses.csv')
    labels = pd.read_csv('../Testbed/labels.csv')

    df = prompts.merge(responses, on='ID').merge(labels, on='ID')

    main_df = df[df['ID'].str.match(fr'^{args.category}-\d+$')].copy().reset_index(drop=True)
    alternate_df = df[df['ID'].str.match(fr'^{args.category}-A-\d+$')].copy().reset_index(drop=True)

    embeddings = OllamaEmbeddings(model='mistral-small')
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    X_main, y_main = make_tensors(main_df, embeddings)
    X_no, y_no = make_tensors(alternate_df, embeddings, label='No')
    X_yes, y_yes = make_tensors(alternate_df, embeddings, label='Yes')

    run_experiment(X_main, y_main, X_no, y_no, device)

    X_combined = torch.cat([X_main, X_no])
    y_combined = torch.cat([y_main, y_no])
    run_experiment(X_combined, y_combined, X_yes, y_yes, device)


if __name__ == '__main__':
    main()