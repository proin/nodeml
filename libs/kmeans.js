'use strict';

module.exports = function () {
    let app = this;

    let assert = (condition, message) => {
        if (!condition) {
            message = message || "Assertion failed";
            if (typeof Error !== "undefined") {
                throw new Error(message);
            }
            throw message;
        }
    };

    let _trained = null;

    app.setModel = (trained) => {
        _trained = trained;
    };

    app.getModel = () => _trained;

	let initialize = (method, k, dataset, labels)=> {
		let center = [];
		if(labels) {
			let map = {}, mapid = 0, r = [];
			for(let i = 0 ; i < labels.length ; i++) {
				if(typeof map[labels[i]] == 'undefined') {
					map[labels[i]] = mapid;
					mapid++;
				}
				r[i] = map[labels[i]];
			}

			k = mapid;
			center = Array.apply(null, Array(k));
			let centerSum = Array.apply(null, Array(k)).map(Number.prototype.valueOf, 0);
			for(let n = 0 ; n < dataset.length ; n++) {
				let _k = r[n] * 1;
				let x = dataset[n];

				if(!center[_k]) {
					center[_k] = {};
				}

				for(let key in x) {
					if(!center[_k][key]) center[_k][key] = 0;
					center[_k][key] += x[key] * 1;
				}

				centerSum[_k]++;
			}

			for(let _k = 0 ; _k < center.length ; _k++)
				for(let _key in center[_k])
					center[_k][_key] = center[_k][_key] / centerSum[_k * 1];
		} else if(method == 'forgy') {
		} else if(method == 'random') {
			let preRand = {};
			while(true) {
				let rand = Math.floor(Math.random() * dataset.length);	
				if(preRand[rand]) continue;
				if(dataset[rand]) {
					center.push(dataset[rand]);
					preRand[rand] = true;
				}
				if(center.length == k) break;
			}
		}

		return center;
	};

	// euclidean distance
	let distance = (x, y)=> {
		let sum = 0;
		let keys = {};
		for(let key in x) keys[key] = true;
		for(let key in y) keys[key] = true;
		
		for(let key in keys) {
			let xd = x[key] ? x[key] * 1 : 0;
			let yd = y[key] ? y[key] * 1 : 0;
			sum += (xd - yd) * (xd - yd);
		}

		return Math.sqrt(sum);
	};

	let em = (center, dataset)=> {
		let r = [];
		for(let n = 0 ; n < dataset.length ; n++) {
			let x = dataset[n];
			let minDist = -1, rn = 0;
			for(let k = 0 ; k < center.length ; k++) {
				let dist = distance(dataset[n], center[k]);			
				if(minDist === -1 || minDist > dist) {
					minDist = dist;
					rn = k;
				}
			}
			r[n] = rn;
		}

		center = Array.apply(null, Array(center.length));
		let centerSum = Array.apply(null, Array(center.length)).map(Number.prototype.valueOf, 0);
		for(let n = 0 ; n < dataset.length ; n++) {
			let k = r[n] * 1;
			let x = dataset[n];
			
			if(!center[k]) center[k] = {};

			for(let key in x) {
				if(!center[k][key]) center[k][key] = 0;
				center[k][key] += x[key] * 1;
			}

			centerSum[k]++;
		}

		for(let k = 0 ; k < center.length ; k++) {
			for(let _key in center[k]) {
				center[k][_key] = center[k][_key] / centerSum[k * 1];
			}
		}

		let J = 0;
		for(let n = 0 ; n < dataset.length ; n++) {
			let x = dataset[n];
			let minDist = -1;
			for(let k = 0 ; k < center.length ; k++) {
				let dist = distance(dataset[n], center[k]);			
				if(minDist === -1 || minDist > dist) {
					minDist = dist;
				}
			}
			J += minDist;
		}

		return {center: center, J: J};
	};

    app.train = (dataset, opts) => {
        assert(dataset, `dataset undefined`);

		// options
		if(!opts) opts = {};
		let { init, dm, proc, iter, labels, k } = opts;
		if(!init) init = 'random'; // initializing method
		if(!dm) dm = 0; // distortion measure threshold
		if(!iter) iter = -1; // maximum iteration
		if(!proc) proc = ()=> {}; // process handler

		// check dataset
        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];

		let n = dataset.length;

		if(!k) k = Math.floor(Math.sqrt(n / 2));
		if(k === 0) k = 1;

		let map = {}, mapid = 0;
		if(labels) {
			for(let i = 0 ; i < labels.length ; i++) {
				if(typeof map[labels[i]] == 'undefined') {
					map[labels[i]] = mapid;
					mapid++;
				}
			}
			
			let tmp = {};
			for(let key in map) 
				tmp[map[key]] = key;
			map = tmp;
		}

		// initialize centre vector
		let center = initialize(init, k, dataset, labels);
		if(center.length == 0) {
			init = 'random';
			center = initialize(init, k, dataset);
		}

		// iteration until J will be the smallest
		let _iter = 0;
		let preJ = 0;
		while(true) {
			if(iter !== -1 && iter < _iter) break;
			let _em = em(center, dataset);
			center = _em.center;
			let J = _em.J;
			let diff = Math.abs(preJ - J);
			proc(_iter, J, diff);
			if(diff <= dm) break;
			preJ = J;
			_iter++;
		}

		_trained = { center: center, map: map };
		return _trained;
    };

    app.test = (dataset) => {
        assert(dataset, `dataset undefined`);
        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];

		let center = _trained.center;
		let map = _trained.map;

		let result = [];
		for(let n = 0 ; n < dataset.length ; n++) {
			let x = dataset[n];
			let minDist = -1, rn = 0;
			for(let k = 0 ; k < center.length ; k++) {
				let dist = distance(dataset[n], center[k]);			
				if(minDist === -1 || minDist > dist) {
					minDist = dist;
					rn = k;
				}
			}
			
			result[n] = map[rn] ? map[rn] : rn;
		}
		return result;
    };

    return app;
};
