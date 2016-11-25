'use strict';

module.exports = function () {
    let app = this;

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

    app.setModel = (trained)=> {
        assert(trained.constructor == Object, `dataset undefined`);
        _trained = trained;
    };

    app.getModel = ()=> _trained;

    app.train = (dataset, labels)=> {
        assert(dataset, `dataset undefined`);
        assert(labels, `labels undefined`);

        if (Array.isArray(dataset) === false) dataset = [dataset];
        if (Array.isArray(labels) === false) labels = [labels];

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

        return _trained;
    };

    app.test = (dataset)=> {
        assert(dataset, `dataset undefined`);
        if (Array.isArray(dataset) && Array.isArray(dataset[0]) === false) dataset = [dataset];
        if (Array.isArray(dataset) === false) dataset = [dataset];

        let _dic = _trained.dic;
        let _label = _trained.label;

        let result = [];

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

            result.push({answer: answer, score: prob});
        }

        if (dataset.length == 1) return result[0];
        else return result;
    };

    return app;
};