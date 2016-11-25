# nodeml

Machine Learning Framework for Node

## How to Use

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

cnn.setModel(layer);

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

## Todo

- Support additional models
    - SVM
    - RNN (LSTM)
    - K-Means
- Evaluation tools
    - calculate precision, recall from result