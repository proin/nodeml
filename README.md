# nodeml

> Machine Learning Framework for Node

## Summary

- Feature Selection
	- `nodeml.feature.tfidf`: [tfidf](https://github.com/proin/nodeml/blob/master/example/tfidf.js)
- Classification
    - `nodeml.Bayes`: [Bayes](https://github.com/proin/nodeml/blob/master/example/nodeml.bayes.js)
    - `nodeml.kNN`: [k-Nearest Neighbor](https://github.com/proin/nodeml/blob/master/example/nodeml.knn.js)
    - `nodeml.CNN`: [Convolutional Neural Network (CNN)](https://github.com/proin/nodeml/blob/master/example/nodeml.cnn.js)
- Clustering
    - `nodeml.kMeans`: [k-Means](https://github.com/proin/nodeml/blob/master/example/nodeml.kmeans.js)
- Recommendation
    - `nodeml.CF`: [User based Collaborative Filtering](https://github.com/proin/nodeml/blob/master/example/nodeml.cf.js)
- Evaluation
    - `nodeml.accuracy`: Precision, Recall, F-Measure, Accuracy
    - `nodeml.ndcg`: NDCG

## Installation

installation on your project

```sh
npm install --save nodeml
```

use example

```js
const {Bayes} = require('nodeml');
let bayes = new Bayes();

bayes.train({'fun': 3, 'couple': 1}, 'comedy');
bayes.train({'couple': 1, 'fast': 1, 'fun': 3}, 'comedy');
bayes.train({'fast': 3, 'furious': 2, 'shoot': 2}, 'action');
bayes.train({'furious': 2, 'shoot': 4, 'fun': 1}, 'action');
bayes.train({'fly': 2, 'fast': 3, 'shoot': 2, 'love': 1}, 'action');

let result = bayes.test({'fun': 3, 'fast': 3, 'shoot': 2});
console.log(result); // this print {answer: , score: }
```

## Document

### nodeml.sample

Sample dataset for test

```js
const {sample} = require('nodeml');

// bbc: Function() => { dataset: [ {} , ... ], labels: [ ... ] }
// bbc news dataset, sparse matrix
const bbc = sample.bbc();

// yeast: Function() => { dataset: [ [] , ... ], labels: [ ... ] }
// yeast dataset, array data
const yeast = sample.yeast();

// iris: Function() => { dataset: [ [] , ... ], labels: [ ... ] }
// iris dataset, array data
const iris = sample.iris();

// movie: Function() => [{ movie_id: '1', user_id: '97', rating: '5', like: '17' }, ...]
// movie dataset, array data
const movie = sample.movie();
```
---

### nodeml.Bayes

Naive Bayes classifier

```js
const {Bayes} = require('nodeml');
let bayes = new Bayes(); // this is bayes classfier
```

#### train: Function(data, label) => model

training bayes classifier

```js
bayes.train([0.2, 0.5, 0.7, 0.4], 1);       
bayes.train({ 'my': 20, 'home': 30 }, 1);   

// training bulk
bayes.train([[2, 5,], [2, 1,]], [1, 2]);    
bayes.train([{}, {}], [1, 2]);              
```

#### test: Function(data) => { answer: string, score: {} }

classify document

```js
let result = bayes.test([2, 5, 1, 4]);
let result = bayes.test({'fun': 3, 'fast': 3, 'shoot': 2});
```

#### getModel: Function () => model

get trained result

```js
let model = bayes.getModel();
let str = JSON.stringify(model);
```

#### setModel: Function (model)

set pre-trained

```js
bayes.setModel(JSON.parse(str));
```

---

### nodeml.kNN

k-Nearest Neighbor Classifier

```js
const {kNN} = require('nodeml');
let knn = new kNN();
```

#### train: Function(dataset, labels) => model

training

```js
knn.train([0.2, 0.5, 0.7, 0.4], 1);       
knn.train({ 'my': 20, 'home': 30 }, 1);   

// training bulk
knn.train([[2, 5,], [2, 1,]], [1, 2]);    
knn.train([{ 'my': 20, 'home': 30 }, { 'my': 5, 'home': 10 }], [1, 2]);              
```

#### test: Function(dataset, k) => [ class1, class2, class1 ]

classify document (default k is 3)

```js
let result = knn.test([2, 5, 1, 4]);
let result = knn.test({'fun': 3, 'fast': 3, 'shoot': 2}, 5);
```

#### getModel: Function () => model

get trained result

```js
let model = knn.getModel();
let str = JSON.stringify(model);
```

#### setModel: Function (model)

set pre-trained

```js
knn.setModel(JSON.parse(str));
```

---

### nodeml.CNN

Convolutional Neural Network, based [convnetjs](http://cs.stanford.edu/people/karpathy/convnetjs)

```js
const {CNN} = require('nodeml');
let cnn = new CNN();
```

#### configure: Function (options)

options object refer `trainer option` at [convnetjs](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html)

```js
cnn.configure({learning_rate: 0.1, momentum: 0.001, batch_size: 5, l2_decay: 0.0001});
```

#### setModel: Function (layer or model)

layer refer at [convnetjs](http://cs.stanford.edu/people/karpathy/convnetjs/docs.html)

```js
var layer = [];
layer.push({type: 'input', out_sx: 1, out_sy: 1, out_depth: 8});
layer.push({type: 'svm', num_classes: 10});

cnn.makeLayer(layer);

// set pre-trained
cnn.setModel(JSON.parse(str));
```

#### train: Function (data, label)

```js
cnn.train([0.2, 0.5, 0.7, 0.4], 1);       
cnn.train({ 'my': 20, 'home': 30 }, 1);   

// training bulk
cnn.train([[2, 5,], [2, 1,]], [1, 2]);    
cnn.train([{}, {}], [1, 2]);   
```

#### test: Function(data) => { answer: string, score: {} }

classify document

```js
let result = cnn.test([2, 5, 1, 4]);
let result = cnn.test({'fun': 3, 'fast': 3, 'shoot': 2});
```

#### getModel: Function () => model

get trained result

```js
let model = cnn.getModel();
let str = JSON.stringify(model);
```
---

### nodeml.kMeans

k-Means Clustering

```js
const {kMeans} = require('nodeml');
let kmeans = new kMeans();
```

#### train: Function(dataset, options) => model

training

```js
kmeans.train([[2, 5,], [2, 1,]], {
    k: 10, dm: 0.00001, iter: 100,  
    proc: (iter, j, d)=> { console.log(iter, j, d); }
});
```

| options | description | type | default |
|---|---|---|---|
| init | cluster initialize function: `random`, `fuzzy (preparing)` | string | 'random' |
| k | number of cluster | integer | 3 |
| dm | distortion measure | float | 0.00 |
| iter | maximum iteration | integer | unlimited |
| labels | supervised learning, if labels exists, detect k automatically | array | null |
| proc | process handler | function | null |

#### test: Function(dataset) => [ class1, class2, class1 ]

classify document (default k is 3)

```js
let result = kmeans.test([[2, 5,], [2, 1,]]);
```

#### getModel: Function () => model

get trained result

```js
let model = kmeans.getModel();
let str = JSON.stringify(model);
```

#### setModel: Function (model)

set pre-trained

```js
kmeans.setModel(JSON.parse(str));
```

---

### nodeml.CF

Collaborative Filtering Function

```js
const {CF, evaluation} = require('../index');

let train = [[1, 1, 2], [1, 2, 2], [1, 4, 5], [2, 3, 2],
    [2, 5, 1], [3, 1, 2], [3, 2, 3], [3, 3, 3]];
let test = [[3, 4, 1]];

const cf = new CF();
cf.train(train);
let gt = cf.gt(test);
let result = cf.recommendGT(gt, 1);

let ndcg = evaluation.ndcg(gt, result);

console.log(gt);
console.log(result);
console.log(ndcg);
```

#### train: Function

---

### nodeml.evaluate

#### accuracy: Function (gt, result) => {precision, recall, f-measure, accuracy}

```js
let {evaluate} = require('nodeml');

let original = [1, 2, 1, 1, 3]; // original label
let result = [1, 1, 2, 1, 3]; // train result label

// exec evaluate, this contains accuracy, micro/macro precision/recall/f-measure
let accuracy = evaluate.accuracy(original, result);
```

#### ndcg: Function (gt, result) => 0 ~ 1 ndcg value

```js
let {CF, evaluate} = require('nodeml');
const cf = new CF();
let gt = cf.gt(test, 'user_id', 'movie_id', 'rating');

let result = cf.recommandToUsers(users, 40);

let ndcg = evaluation.ndcg(gt, result);
```
