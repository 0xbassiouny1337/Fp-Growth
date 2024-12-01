import pandas as pd
from collections import defaultdict
from itertools import combinations
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate

class FPTreeNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.link = None  # Link to the next node of the same item

    def increment(self, count):
        self.count += count

class FPTree:
    def __init__(self, transactions, min_support):
        self.min_support = min_support
        self.header_table = {}
        self.root = FPTreeNode(None, 1, None)
        self.build_tree(transactions)

    def build_tree(self, transactions):
        freq = defaultdict(int)
        for transaction in transactions:
            for item in transaction:
                freq[item] += 1
        print("Item Frequencies:", freq)  # Debugging output

        freq = {k: v for k, v in freq.items() if v >= self.min_support}
        print("Filtered Frequencies:", freq)  # Debugging output

        if not freq:
            return

        for item in freq:
            self.header_table[item] = None

        for transaction in transactions:
            filtered_transaction = [item for item in transaction if item in freq]
            filtered_transaction.sort(key=lambda x: freq[x], reverse=True)
            self.insert_tree(filtered_transaction, self.root)

    def insert_tree(self, transaction, node):
        if not transaction:
            return

        first_item = transaction[0]
        if first_item in node.children:
            node.children[first_item].increment(1)
        else:
            new_node = FPTreeNode(first_item, 1, node)
            node.children[first_item] = new_node

            if not self.header_table[first_item]:
                self.header_table[first_item] = new_node
            else:
                current = self.header_table[first_item]
                while current.link:
                    current = current.link
                current.link = new_node

        self.insert_tree(transaction[1:], node.children[first_item])

    def mine_patterns(self):
        patterns = {}
        for item in sorted(self.header_table.keys(), key=lambda x: x, reverse=True):
            item_support = sum(node.count for node in self.iterate_nodes(item))
            if item_support >= self.min_support:
                patterns[frozenset([item])] = item_support
            patterns.update(self.mine_subtree(item))
        return patterns

    def mine_subtree(self, item):
        conditional_patterns = []
        node = self.header_table[item]
        while node:
            prefix_path = []
            current = node.parent
            while current and current.item is not None:
                prefix_path.append(current.item)
                current = current.parent
            if prefix_path:
                conditional_patterns.extend([prefix_path[::-1]] * node.count)
            node = node.link

        if not conditional_patterns:
            return {}

        conditional_tree = FPTree(conditional_patterns, self.min_support)
        sub_patterns = conditional_tree.mine_patterns()

        patterns = {}
        for pattern, support in sub_patterns.items():
            combined_pattern = frozenset(list(pattern) + [item])
            patterns[combined_pattern] = support

        return patterns

    def iterate_nodes(self, item):
        node = self.header_table.get(item)
        while node:
            yield node
            node = node.link

    def visualize_tree(self):
        G = nx.DiGraph()
        node_labels = {}

        def add_edges(node, parent_label):
            node_label = f"{node.item} ({node.count})" if node.item else "Root"
            node_labels[node_label] = node.item if node.item else "Root"

            if parent_label:
                G.add_edge(parent_label, node_label)

            for child in node.children.values():
                add_edges(child, node_label)

        add_edges(self.root, None)

        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=3000, font_size=10, node_color="lightblue")
        plt.title("FP-Tree Visualization")
        plt.show()


def load_data(filepath):
    df = pd.read_excel(filepath)
    transactions = df['items'].apply(lambda x: x.split(',')).tolist()
    return [list(map(str.strip, transaction)) for transaction in transactions]

def generate_association_rules(frequent_itemsets, min_confidence):
    rules = []
    for itemset, support in frequent_itemsets.items():
        if len(itemset) > 1:
            for i in range(1, len(itemset)):
                for subset in combinations(itemset, i):
                    antecedent = frozenset(subset)
                    consequent = itemset - antecedent
                    antecedent_support = frequent_itemsets.get(antecedent, 0)
                    if antecedent_support > 0:
                        confidence = support / antecedent_support
                        if confidence >= min_confidence:
                            rules.append((antecedent, consequent, confidence))
    return rules

def calculate_lift(rules, frequent_itemsets, total_transactions):
    lifts = []
    for antecedent, consequent, confidence in rules:
        antecedent_support = frequent_itemsets[frozenset(antecedent)] / total_transactions
        consequent_support = frequent_itemsets[frozenset(consequent)] / total_transactions
        lift = confidence / consequent_support
        lifts.append((antecedent, consequent, confidence, lift))
    return lifts

# Main program
file_path = "Horizontal_Format.xlsx"  # Update with the correct path to your Excel file
transactions = load_data(file_path)
print("Transactions:", transactions)  # Debugging output

min_support = 2
min_confidence = 0.7

# Build FP-Tree
fp_tree = FPTree(transactions, min_support)
fp_tree.visualize_tree()

frequent_itemsets = fp_tree.mine_patterns()

# Display Frequent Itemsets in Table
itemsets_table = [["Itemset", "Support"]] + [[set(item), support] for item, support in frequent_itemsets.items()]
print(tabulate(itemsets_table, headers="firstrow", tablefmt="fancy_grid"))

# Generate Association Rules
rules = generate_association_rules(frequent_itemsets, min_confidence)

# Display Rules in Table
rules_table = [["Antecedent", "Consequent", "Confidence"]] + [
    [set(antecedent), set(consequent), confidence] for antecedent, consequent, confidence in rules
]
print(tabulate(rules_table, headers="firstrow", tablefmt="fancy_grid"))

# Calculate Lift
total_transactions = len(transactions)
strong_rules = calculate_lift(rules, frequent_itemsets, total_transactions)

# Display Strong Rules with Lift
lifts_table = [["Antecedent", "Consequent", "Confidence", "Lift"]] + [
    [set(antecedent), set(consequent), confidence, lift] for antecedent, consequent, confidence, lift in strong_rules
]
print(tabulate(lifts_table, headers="firstrow", tablefmt="fancy_grid"))
