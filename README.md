# Adaptive-Compression-through-PPM---4-
This project presents a complete implementation and comparison of two lossless compression algorithms—Adaptive Huffman Coding (AHC) and Prediction by Partial Matching (PPM-C, Order-4)—applied to a large smart home energy consumption dataset. The objective is to evaluate whether multi-order context modeling can significantly improve compression performance for IoT time-series data compared to traditional symbol-frequency–based methods.

Project Overview

Modern smart home systems generate continuous streams of high-frequency energy usage data from smart meters and sensors. These time-series datasets grow quickly and contain substantial redundancy, leading to challenges in storage, bandwidth, and edge processing efficiency.

This project builds a compression framework that:

Implements Adaptive Huffman Coding based on the FGK algorithm

Implements PPM-C (Order-4) with a custom Range Encoder and Decoder

Evaluates both algorithms on the same IoT dataset

Measures compression ratio, space saved, compression time, and decompression time

Produces detailed comparison graphs for analysis

Algorithms Used
1. Adaptive Huffman Coding (Current Method)

Adaptive Huffman Coding updates symbol frequencies dynamically as data is processed. It does not require a predefined frequency table and is suitable for streaming contexts. However, it models only individual symbols and does not capture relationships between consecutive values. As a result, its compression effectiveness is limited for highly structured time-series data.
Due to computational overhead on large datasets, it is applied to the first 0.5 MB of data.

2. PPM-C (Order-4) — Proposed Method

PPM-C is a context-based statistical modeling approach. The Order-4 model predicts each symbol based on the four preceding symbols, enabling it to learn repetitive or periodic usage patterns within smart home energy data.
The model incorporates escape handling and uses a range encoder for representing cumulative probabilities. This allows PPM-4 to achieve significantly better compression while remaining fully lossless.

Key Findings
Algorithm	Compression Ratio	Space Saved (%)	Lossless	Notes
Adaptive Huffman	0.636	36.36%	Yes	Limited context modeling
PPM-4	0.149	85.15%	Yes	Best overall performance

A lower compression ratio indicates better compression.
PPM-4 provides the highest reduction due to its ability to leverage multi-symbol context.
