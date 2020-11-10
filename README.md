# Graph-to-Graph Transformer
Pytorch implementation of ["Graph-to-Graph Transformer for Transition-based Dependency Parsing"](https://www.aclweb.org/anthology/2020.findings-emnlp.294/)

## Sentence Transformer

To reproduce results of SentTr and SentTr+G2GTr model, you can find all required materials in `senttr` directory.

## State Transformer

To reproduce results of all variations of StateTr model, you can find all required materials in `statetr` directory.

## General Graph-to-Graph Transformer

To use our Graph-to-Graph Transformer for other NLP tasks, plese refer to [this repository](https://github.com/idiap/g2g-transformer).  

## Citation

If you use the code for your research, please cite this work as:

```
@inproceedings{mohammadshahi-henderson-2020-graph,
    title = "Graph-to-Graph Transformer for Transition-based Dependency Parsing",
    author = "Mohammadshahi, Alireza  and
      Henderson, James",
    booktitle = "Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: Findings",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.findings-emnlp.294",
    pages = "3278--3289",
    abstract = "We propose the Graph2Graph Transformer architecture for conditioning on and predicting arbitrary graphs, and apply it to the challenging task of transition-based dependency parsing. After proposing two novel Transformer models of transition-based dependency parsing as strong baselines, we show that adding the proposed mechanisms for conditioning on and predicting graphs of Graph2Graph Transformer results in significant improvements, both with and without BERT pre-training. The novel baselines and their integration with Graph2Graph Transformer significantly outperform the state-of-the-art in traditional transition-based dependency parsing on both English Penn Treebank, and 13 languages of Universal Dependencies Treebanks. Graph2Graph Transformer can be integrated with many previous structured prediction methods, making it easy to apply to a wide range of NLP tasks.",
}
```
