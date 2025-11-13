# compression/adaptive_huffman.py
# Optimized Adaptive Huffman (FGK Algorithm) - Byte-level implementation

class Node:
    def __init__(self, symbol=None, weight=0, parent=None, left=None, right=None, number=0):
        self.symbol = symbol
        self.weight = weight
        self.parent = parent
        self.left = left
        self.right = right
        self.number = number

    def is_leaf(self):
        return self.left is None and self.right is None


class AdaptiveHuffman:
    def __init__(self):
        # NYT (Not Yet Transmitted)
        self.NYT = Node(symbol=None, weight=0, number=512)
        self.root = self.NYT
        self.nodes = {None: self.NYT}
        self.node_list = [self.NYT]  

    # ---------- Helpers ----------
    def _find_node(self, symbol):
        return self.nodes.get(symbol)

    def _get_code(self, symbol):
        node = self._find_node(symbol)
        if not node:
            node = self.NYT
        code = ""
        while node.parent:
            code = ("0" if node.parent.left == node else "1") + code
            node = node.parent
        return code

    def _swap_nodes(self, a, b):
        
        if a is b or a.parent is b or b.parent is a:
            return  

        a_parent, b_parent = a.parent, b.parent
        if a_parent.left == a:
            a_parent.left = b
        else:
            a_parent.right = b

        if b_parent.left == b:
            b_parent.left = a
        else:
            b_parent.right = a

        a.parent, b.parent = b_parent, a_parent
        a.number, b.number = b.number, a.number

    def _update_tree(self, node):
        while node:
            # find highest-numbered node with smaller weight
            same_weight = [
                n for n in self.node_list if n.weight == node.weight and n.number > node.number
            ]
            if same_weight:
                swap_node = max(same_weight, key=lambda x: x.number)
                self._swap_nodes(node, swap_node)

            node.weight += 1
            node = node.parent

    def _add_symbol(self, symbol):
        """Add a new symbol to the tree under NYT."""
        new_internal = Node(weight=0, parent=self.NYT.parent, number=self.NYT.number - 2)
        new_leaf = Node(symbol=symbol, weight=0, parent=new_internal, number=self.NYT.number - 1)
        new_internal.left = self.NYT
        new_internal.right = new_leaf

        if self.NYT.parent:
            if self.NYT.parent.left == self.NYT:
                self.NYT.parent.left = new_internal
            else:
                self.NYT.parent.right = new_internal
        else:
            self.root = new_internal

        self.NYT.parent = new_internal
        self.nodes[symbol] = new_leaf
        self.node_list.extend([new_internal, new_leaf])

        return new_leaf

    # ---------- Compression ----------
    def compress_bytes(self, data: bytes) -> bytes:
        output_bits = ""
        for i, symbol in enumerate(data):
            node = self._find_node(symbol)
            if node:
                code = self._get_code(symbol)
                output_bits += code
            else:
                code = self._get_code(None)
                output_bits += code
                output_bits += f"{symbol:08b}"
                node = self._add_symbol(symbol)
            self._update_tree(node)

            if i % 100000 == 0 and i > 0:
                print(f"🔹 Processed {i} bytes...")

        # pad to byte boundary
        while len(output_bits) % 8 != 0:
            output_bits += "0"

        out_bytes = bytes(
            int(output_bits[i:i + 8], 2) for i in range(0, len(output_bits), 8)
        )
        return out_bytes

    # ---------- Decompression ----------
    def decompress_bytes(self, blob: bytes) -> bytes:
        bitstream = "".join(f"{b:08b}" for b in blob)
        result = bytearray()
        node = self.root
        i = 0
        while i < len(bitstream):
            if node.is_leaf():
                if node is self.NYT:
                    if i + 8 > len(bitstream):
                        break
                    sym = int(bitstream[i:i + 8], 2)
                    i += 8
                    result.append(sym)
                    node = self._add_symbol(sym)
                    self._update_tree(node)
                    node = self.root
                else:
                    sym = node.symbol
                    result.append(sym)
                    self._update_tree(node)
                    node = self.root
            else:
                bit = bitstream[i]
                i += 1
                node = node.left if bit == "0" else node.right

        return bytes(result)

