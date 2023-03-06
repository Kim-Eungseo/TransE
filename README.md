# TransE
TransE python code

# TransE

[Papers with Code - TransE Explained](https://paperswithcode.com/method/transe)

## TransE의 특징 (검증 후)

1. TransE (Translation-based Embedding) is a popular approach in knowledge graph embedding.
    
    → 암튼 Knowledge graph embedding에서는 유명한 애
    
2. It models entities and relationships as vectors in a low-dimensional space. 
    
    → 벡터 방식으로 모델링한다는 것이 특징
    
    → 차원수는 작음 → 왜 이런 방식을 택했어야 하는가?
    
    1. **Efficiency**: 지식 그래프를 구현하는 방법에 여러가지가 있었는데, TransE는 low-dimensional space에 간단한 전제를 이용하여 성능과 간단함을 모두 얻을 수 있었다.
    2. **Flexibility**: 복잡한 형태의 관계를 표현할 수 있었다.
    3. **Interpretability**: 해석 가능하고 설명이 가능하다는 장점이 있다.
3. TransE represents entities and relationships as vectors in the same vector space.
    
    → 개체와 연결관계를 같은 벡터공간에 벡터로 저장한다.
    
4. It views relationships as translations between entity vectors. 
    
    → 이 논문에서 연결관계를 개체간 “translation”으로 표현하였다.
    
5. The model defines a scoring function to determine the similarity between entities and relationships.
    
    → 개체와 연결관계 간의 유사성을 측정하기 위한 Scoring 함수가 있다.
    
6. TransE uses a margin-based loss function to train the model.
    
    → 말 그대로다. Entity와 relation에 해당하는 값을 추정하기 위해 margin-based loss function을 쓴다.
    
7. The vectors for entities and relationships are optimized to minimize the loss.
8. The embedding space can be **Euclidean** or **Hyperbolic.**
    1. **Euclidean embedding space** is a type of embedding space used in machine learning models that represent data as points or vectors in a multi-dimensional space. In Euclidean embedding space, the distance between two points is measured using the Euclidean distance metric.
        
        More specifically, in Euclidean embedding space, each entity or object is represented as a vector of real numbers, where each dimension of the vector corresponds to a feature or attribute of the object. For example, in the context of knowledge graph embeddings, each entity and relationship is represented as a vector in a low-dimensional space, where the dimensions of the vector correspond to the different properties or attributes of the entity or relationship.
        
        Once the entities and relationships are represented as vectors in the Euclidean embedding space, machine learning models can learn to make predictions by analyzing the relationships between the vectors. For example, in the case of knowledge graph embeddings, models like TransE use vector arithmetic to learn the relationships between entities and predict missing relationships based on the positions of the vectors in the embedding space.
        
        The use of Euclidean embedding space allows for efficient computation and powerful representations of complex relationships between entities. However, it also has some limitations. For example, Euclidean embedding space assumes that the relationships between entities are linear and that the distance between two points is a meaningful measure of similarity. In some cases, more complex embedding spaces like hyperbolic or spherical embedding spaces may be more appropriate for representing certain types of data or relationships. (**ChatGPT**)
        
    2. **Hyperbolic embedding space** is a type of embedding space used in machine learning models that represent data as points or vectors in a non-Euclidean space. In hyperbolic embedding space, the distance between two points is measured using a non-Euclidean distance metric known as the hyperbolic distance.
        
        [Hyperbolic Embedding에 관한 짧은 설명과 고찰- (1)](https://chumji.tistory.com/3)
        
        **The hyperbolic space has some unique properties that make it well-suited for modeling hierarchical or tree-like structures, which are common in many types of data.** In contrast to Euclidean space, hyperbolic space is negatively curved, which means that as we move away from the origin, space expands at an exponential rate. This exponential expansion allows for the representation of hierarchies in a more compact and efficient way than Euclidean space.
        
        In the context of knowledge graph embeddings, hyperbolic embeddings have been shown to be effective at modeling hierarchies and capturing complex relationships between entities. For example, models like **Hyperbolic Knowledge Graph Embeddings (HypE)** and **Hyperbolic TransE (HypTransE)** use hyperbolic embeddings to learn hierarchical representations of knowledge graphs and make predictions based on the positions of the entities in the hyperbolic space.
        
        Overall, hyperbolic embedding space provides a powerful tool for modeling hierarchical structures and capturing complex relationships between entities. Its ability to represent hierarchies more efficiently than Euclidean space makes it well-suited for many types of data, including knowledge graphs. (ChatGPT)
        
9. Asymmetric한 관계, composition의 관계와 inverse의 관계에 대해서 학습이 가능하다.
10. 그래프 관계형 데이터의 어려운 점은 Locality에서 많이 비롯된다.
    
    **Locality**: KG에서 특정 객체는 연결관계가 많아서 비교적 특이하고 적은 양의 객체나 연결관계 종류는 모델에 반영이 잘 안되는 문제를 말함.
    
    → 일반적으로 Locality를 해결하기 위해 weight function을 두거나 외부데이터를 추가하는 방식으로 해결
    
11. 당시에 Expressivity와 Universality 문제를 해결하는 데에 관심이 많았다고 한다.
12. 모델을 단순하게 표현하지 않았을 때, non-convex optimization 문제나 성능 저하 문제가 있었다고 한다.
13. Hierarchical relationship과 이를 표현하는 translation을 전제로 만든 모델
    
    → KB 자체의 특성을 잘 찾는 것이 중요해 보인다.
    
    → 근데 Hierarchies를 모델링하기 위해 만들었는데 relationship 종류 상관없이 잘 먹히더라.
    
    → 다른 구조들은 뭐가 있는가?
    
14. TransE는 symmetric한 relation과 one-to-many relation에 대해서는 주어진 scoring function에서는 학습이 어렵다.

---

## Learning에 관하여

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
위 알고리즘의 1, 2, 3번 행에서는 주어진 범위 사이의 랜덤한 값으로 e(entity)와 l(relation)에 부여했다.

6번 줄에서는 주어진 데이터의 triplet들을 모아둔 set을 일정한 batch로 나누었다.

8~11번 줄의 반복문에서는 batch 안의 모든 triplet에 대해서 다음의 과정을 통해 corrupted set을 만든다.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
마지막으로는, Stocastic gradient descent을 통해 다음 loss function의 최솟값을 추정한다.

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/image.jpg?raw=true)
---

## 의견

- Word2vec와 상당히 유사한 흐름을 보여주는 것 같다. 단어 간 거리를 연구하는 것과 entity 간 거리를 relation으로 설명하려고 하는 시도는 데이터를 백터화하여 learning을 수행했다는 점에서 같아 보인다.
- 이 모델에서 locality 문제를 해결하고자 loss function에 weight값을 넣은 것이 아닌가 싶다.
- 제프리 힌튼의 F-F 알고리즘을 사용할 수는 없을까?
- Transformer처럼 극단적인 병렬화를 해낼 수는 없을까?
- Category theory에서 말하는 구조들에서 더 나은 전제를 제공할 순 없을까?
