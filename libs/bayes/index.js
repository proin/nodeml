'use strict';

module.exports = ()=> {
    let app = {};

    let assert = (condition, message)=> {
        if (!condition) {
            message = message || "Assertion failed";
            if (typeof Error !== "undefined") {
                throw new Error(message);
            }
            throw message;
        }
    };

    let _trained = null;

    app.setTrain = (trained)=> new Promise((resolve)=> {
        assert(trained.constructor == Object, `dataset undefined`);
        _trained = trained;
        resolve();
    });

    app.getTrain = ()=> new Promise((resolve)=> {
        resolve(_trained);
    });

    app.saveTrain = ()=> new Promise((resolve)=> {
        resolve();
    });

    app.train = (dataset, labels)=> new Promise((resolve)=> {
        assert(dataset, `dataset undefined`);
        assert(labels, `labels undefined`);
        if (dataset[0].constructor != Array && dataset[0].constructor != Object) dataset = [dataset];
        if (labels.constructor != Array) labels = [labels];
        assert(dataset.length === labels.length, `mismatched array length`);

        if (!_trained) _trained = {};
        if (!_trained.dic) _trained.dic = {};
        if (!_trained.label) _trained.label = {};
        let _dic = _trained.dic;
        let _label = _trained.label;

        for (let i = 0; i < dataset.length; i++) {
            let data = dataset[i];
            let label = labels[i];

            let sum = 0;
            for (let key in data) {
                if (!_dic[key]) _dic[key] = {};
                if (!_dic[key][label]) _dic[key][label] = 0;
                _dic[key][label] += data[key] * 1;
                sum += data[key] * 1;
            }

            if (!_label[label]) _label[label] = 0;
            _label[label] += sum;
        }

        resolve(_trained);
    });

    app.test = (dataset, opts)=> new Promise((resolve)=> {
        assert(dataset, `dataset undefined`);
        if (dataset.constructor != Array) dataset = [dataset];

        let labels = opts.labels;
        let probability = opts.prob;
        if (labels && labels.constructor != Array) labels = [labels];

        let _dic = _trained.dic;
        let _label = _trained.label;

        let precision = 0;

        let result = {};
        result.answer = [];
        if (probability) result.prob = [];

        for (let i = 0; i < dataset.length; i++) {
            let prob = {};
            let data = dataset[i];
            for (let label in _label) {
                if (typeof prob[label] === 'undefined') prob[label] = 0;
                for (let key in data) {
                    let fc = _dic[key] ? _dic[key][label] ? _dic[key][label] * data[key] : 0 : 0;
                    fc += 1;
                    prob[label] += Math.log(fc / _label[label]);
                }
            }

            let max = null;
            let answer = null;

            for (let label in prob) {
                if (!max) {
                    max = prob[label];
                    answer = label;
                }

                if (prob[label] > max) {
                    max = prob[label];
                    answer = label;
                }
            }

            result.answer.push(answer);

            if (probability) result.prob.push(prob);

            if (labels && labels[i] && labels[i] == answer)
                precision++;
        }

        if (labels) result.precision = precision / dataset.length;
        resolve(result);
    });

    return app;
};