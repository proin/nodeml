'use strict';

const fs = require('fs');

module.exports = function () {
    let app = this;

    // utilities
    let return_v = false;
    let v_val = 0.0;
    let gaussRandom = () => {
        if (return_v) {
            return_v = false;
            return v_val;
        }
        let u = 2 * Math.random() - 1;
        let v = 2 * Math.random() - 1;
        let r = u * u + v * v;
        if (r == 0 || r > 1) return gaussRandom();
        let c = Math.sqrt(-2 * Math.log(r) / r);
        v_val = v * c; // cache this
        return_v = true;
        return u * c;
    };
    let randf = (a, b) => Math.random() * (b - a) + a;
    let randi = (a, b) => Math.floor(Math.random() * (b - a) + a);
    let randn = (mu, std) => mu + gaussRandom() * std;
    let zeros = (n) => {
        if (typeof(n) === 'undefined' || isNaN(n)) return [];

        if (typeof ArrayBuffer === 'undefined') {
            let arr = new Array(n);
            for (var i = 0; i < n; i++)
                arr[i] = 0;
            return arr;
        }

        return new Float64Array(n);
    };
    let arrContains = (arr, elt) => {
        for (let i = 0, n = arr.length; i < n; i++)
            if (arr[i] === elt)
                return true;
        return false;
    };
    let arrUnique = (arr) => {
        var b = [];
        for (let i = 0, n = arr.length; i < n; i++)
            if (!arrContains(b, arr[i]))
                b.push(arr[i]);
        return b;
    };
    let maxmin = (w) => {
        if (w.length === 0)
            return {};
        let maxv = w[0];
        let minv = w[0];
        let maxi = 0;
        let mini = 0;
        let n = w.length;
        for (let i = 1; i < n; i++) {
            if (w[i] > maxv) {
                maxv = w[i];
                maxi = i;
            }
            if (w[i] < minv) {
                minv = w[i];
                mini = i;
            }
        }
        return {maxi: maxi, maxv: maxv, mini: mini, minv: minv, dv: maxv - minv};
    };
    let randperm = (n) => {
        let i = n;
        let j = 0;
        let temp = null;
        let array = [];
        for (let q = 0; q < n; q++) array[q] = q;
        while (i--) {
            j = Math.floor(Math.random() * (i + 1));
            temp = array[i];
            array[i] = array[j];
            array[j] = temp;
        }
        return array;
    };
    let weightedSample = (lst, probs) => {
        let p = randf(0, 1.0);
        let cumprob = 0.0;
        for (let k = 0, n = lst.length; k < n; k++) {
            cumprob += probs[k];
            if (p < cumprob)
                return lst[k];
        }
    };
    let getopt = (opt, field_name, default_value) => typeof opt[field_name] !== 'undefined' ? opt[field_name] : default_value;

    // Class: Vol
    let Vol = function (sx, sy, depth, c) {
        this.constructor = 'CnnVolume';
        if (Object.prototype.toString.call(sx) === '[object Array]') {
            this.sx = 1;
            this.sy = 1;
            this.depth = sx.length;
            this.w = zeros(this.depth);
            this.dw = zeros(this.depth);
            for (let i = 0; i < this.depth; i++)
                this.w[i] = sx[i];
        } else {
            this.sx = sx;
            this.sy = sy;
            this.depth = depth;
            let n = sx * sy * depth;
            this.w = zeros(n);
            this.dw = zeros(n);
            if (typeof c === 'undefined') {
                let scale = Math.sqrt(1.0 / (sx * sy * depth));
                for (let i = 0; i < n; i++)
                    this.w[i] = randn(0.0, scale);
            } else {
                for (let i = 0; i < n; i++)
                    this.w[i] = c;
            }
        }
    };
    Vol.prototype = {
        get: function (x, y, d) {
            let ix = ((this.sx * y) + x) * this.depth + d;
            return this.w[ix];
        },
        set: function (x, y, d, v) {
            let ix = ((this.sx * y) + x) * this.depth + d;
            this.w[ix] = v;
        },
        add: function (x, y, d, v) {
            let ix = ((this.sx * y) + x) * this.depth + d;
            this.w[ix] += v;
        },
        get_grad: function (x, y, d) {
            let ix = ((this.sx * y) + x) * this.depth + d;
            return this.dw[ix];
        },
        set_grad: function (x, y, d, v) {
            let ix = ((this.sx * y) + x) * this.depth + d;
            this.dw[ix] = v;
        },
        add_grad: function (x, y, d, v) {
            let ix = ((this.sx * y) + x) * this.depth + d;
            this.dw[ix] += v;
        },
        cloneAndZero: function () {
            return new Vol(this.sx, this.sy, this.depth, 0.0)
        },
        clone: function () {
            let V = new Vol(this.sx, this.sy, this.depth, 0.0);
            let n = this.w.length;
            for (let i = 0; i < n; i++) {
                V.w[i] = this.w[i];
            }
            return V;
        },
        addFrom: function (V) {
            for (let k = 0; k < this.w.length; k++) {
                this.w[k] += V.w[k];
            }
        },
        addFromScaled: function (V, a) {
            for (let k = 0; k < this.w.length; k++) {
                this.w[k] += a * V.w[k];
            }
        },
        setConst: function (a) {
            for (let k = 0; k < this.w.length; k++) {
                this.w[k] = a;
            }
        },
        toJSON: function () {
            let json = {};
            json.sx = this.sx;
            json.sy = this.sy;
            json.depth = this.depth;
            json.w = this.w;
            return json;
        },
        fromJSON: function (json) {
            this.sx = json.sx;
            this.sy = json.sy;
            this.depth = json.depth;
            let n = this.sx * this.sy * this.depth;
            this.w = zeros(n);
            this.dw = zeros(n);
            for (let i = 0; i < n; i++) {
                this.w[i] = json.w[i];
            }
        }
    };

    // Volume utilities
    let augment = function (V, crop, dx, dy, fliplr) {
        if (!fliplr) fliplr = false;
        if (!dx) dx = randi(0, V.sx - crop);
        if (!dy) dy = randi(0, V.sy - crop);
        let W = null;
        if (crop !== V.sx || dx !== 0 || dy !== 0) {
            W = new Vol(crop, crop, V.depth, 0.0);
            for (let x = 0; x < crop; x++) {
                for (let y = 0; y < crop; y++) {
                    if (x + dx < 0 || x + dx >= V.sx || y + dy < 0 || y + dy >= V.sy) continue;
                    for (let d = 0; d < V.depth; d++)
                        W.set(x, y, d, V.get(x + dx, y + dy, d));
                }
            }
        } else {
            W = V;
        }

        if (fliplr) {
            let W2 = W.cloneAndZero();
            for (let x = 0; x < W.sx; x++)
                for (let y = 0; y < W.sy; y++)
                    for (let d = 0; d < W.depth; d++)
                        W2.set(x, y, d, W.get(W.sx - x - 1, y, d));
            W = W2;
        }

        return W;
    };

    // set Volume
    app.Vol = Vol;
    app.augment = augment;

    // Class: Layers
    let ConvLayer = function (opt) {
        opt = opt || {};

        this.out_depth = opt.filters;
        this.sx = opt.sx;
        this.in_depth = opt.in_depth;
        this.in_sx = opt.in_sx;
        this.in_sy = opt.in_sy;

        this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
        this.stride = typeof opt.stride !== 'undefined' ? opt.stride : 1;
        this.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0;
        this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
        this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;

        this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.layer_type = 'conv';

        let bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
        this.filters = [];
        for (let i = 0; i < this.out_depth; i++)
            this.filters.push(new Vol(this.sx, this.sy, this.in_depth));
        this.biases = new Vol(1, 1, this.out_depth, bias);
    };
    ConvLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
            for (let d = 0; d < this.out_depth; d++) {
                let f = this.filters[d];
                let x = -this.pad;
                let y = -this.pad;
                for (let ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                    y = -this.pad;
                    for (let ay = 0; ay < this.out_sy; y += this.stride, ay++) {
                        let a = 0.0;
                        for (let fx = 0; fx < f.sx; fx++) {
                            for (let fy = 0; fy < f.sy; fy++) {
                                for (let fd = 0; fd < f.depth; fd++) {
                                    let oy = y + fy;
                                    let ox = x + fx;
                                    if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx)
                                        a += f.w[((f.sx * fy) + fx) * f.depth + fd] * V.w[((V.sx * oy) + ox) * V.depth + fd];
                                }
                            }
                        }
                        a += this.biases.w[d];
                        A.set(ax, ay, d, a);
                    }
                }
            }
            this.out_act = A;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            V.dw = zeros(V.w.length);
            for (let d = 0; d < this.out_depth; d++) {
                let f = this.filters[d];
                let x = -this.pad;
                let y = -this.pad;
                for (let ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                    y = -this.pad;
                    for (let ay = 0; ay < this.out_sy; y += this.stride, ay++) {
                        let chain_grad = this.out_act.get_grad(ax, ay, d);
                        for (let fx = 0; fx < f.sx; fx++) {
                            for (let fy = 0; fy < f.sy; fy++) {
                                for (let fd = 0; fd < f.depth; fd++) {
                                    let oy = y + fy;
                                    let ox = x + fx;
                                    if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                                        let ix1 = ((V.sx * oy) + ox) * V.depth + fd;
                                        let ix2 = ((f.sx * fy) + fx) * f.depth + fd;
                                        f.dw[ix2] += V.w[ix1] * chain_grad;
                                        V.dw[ix1] += f.w[ix2] * chain_grad;
                                    }
                                }
                            }
                        }
                        this.biases.dw[d] += chain_grad;
                    }
                }
            }
        },
        getParamsAndGrads: function () {
            let response = [];
            for (let i = 0; i < this.out_depth; i++)
                response.push({params: this.filters[i].w, grads: this.filters[i].dw, l2_decay_mul: this.l2_decay_mul, l1_decay_mul: this.l1_decay_mul});
            response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
            return response;
        },
        toJSON: function () {
            let json = {};
            json.sx = this.sx;
            json.sy = this.sy;
            json.stride = this.stride;
            json.in_depth = this.in_depth;
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            json.l1_decay_mul = this.l1_decay_mul;
            json.l2_decay_mul = this.l2_decay_mul;
            json.pad = this.pad;
            json.filters = [];
            for (let i = 0; i < this.filters.length; i++)
                json.filters.push(this.filters[i].toJSON());
            json.biases = this.biases.toJSON();
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
            this.sx = json.sx;
            this.sy = json.sy;
            this.stride = json.stride;
            this.in_depth = json.in_depth;
            this.filters = [];
            this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
            this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
            this.pad = typeof json.pad !== 'undefined' ? json.pad : 0;
            for (let i = 0; i < json.filters.length; i++) {
                let v = new Vol(0, 0, 0, 0);
                v.fromJSON(json.filters[i]);
                this.filters.push(v);
            }
            this.biases = new Vol(0, 0, 0, 0);
            this.biases.fromJSON(json.biases);
        }
    };

    let FullyConnLayer = function (opt) {
        opt = opt || {};
        this.out_depth = typeof opt.num_neurons !== 'undefined' ? opt.num_neurons : opt.filters;
        this.l1_decay_mul = typeof opt.l1_decay_mul !== 'undefined' ? opt.l1_decay_mul : 0.0;
        this.l2_decay_mul = typeof opt.l2_decay_mul !== 'undefined' ? opt.l2_decay_mul : 1.0;
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = 'fc';

        let bias = typeof opt.bias_pref !== 'undefined' ? opt.bias_pref : 0.0;
        this.filters = [];
        for (let i = 0; i < this.out_depth; i++)
            this.filters.push(new Vol(1, 1, this.num_inputs));
        this.biases = new Vol(1, 1, this.out_depth, bias);
    };
    FullyConnLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let A = new Vol(1, 1, this.out_depth, 0.0);
            let Vw = V.w;
            for (let i = 0; i < this.out_depth; i++) {
                let a = 0.0;
                let wi = this.filters[i].w;
                for (let d = 0; d < this.num_inputs; d++)
                    a += Vw[d] * wi[d];
                a += this.biases.w[i];
                A.w[i] = a;
            }
            this.out_act = A;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            V.dw = zeros(V.w.length);
            for (let i = 0; i < this.out_depth; i++) {
                let tfi = this.filters[i];
                let chain_grad = this.out_act.dw[i];
                for (let d = 0; d < this.num_inputs; d++) {
                    V.dw[d] += tfi.w[d] * chain_grad;
                    tfi.dw[d] += V.w[d] * chain_grad;
                }
                this.biases.dw[i] += chain_grad;
            }
        },
        getParamsAndGrads: function () {
            let response = [];
            for (let i = 0; i < this.out_depth; i++)
                response.push({params: this.filters[i].w, grads: this.filters[i].dw, l1_decay_mul: this.l1_decay_mul, l2_decay_mul: this.l2_decay_mul});
            response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
            return response;
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            json.num_inputs = this.num_inputs;
            json.l1_decay_mul = this.l1_decay_mul;
            json.l2_decay_mul = this.l2_decay_mul;
            json.filters = [];
            for (let i = 0; i < this.filters.length; i++)
                json.filters.push(this.filters[i].toJSON());
            json.biases = this.biases.toJSON();
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
            this.num_inputs = json.num_inputs;
            this.l1_decay_mul = typeof json.l1_decay_mul !== 'undefined' ? json.l1_decay_mul : 1.0;
            this.l2_decay_mul = typeof json.l2_decay_mul !== 'undefined' ? json.l2_decay_mul : 1.0;
            this.filters = [];
            for (let i = 0; i < json.filters.length; i++) {
                let v = new Vol(0, 0, 0, 0);
                v.fromJSON(json.filters[i]);
                this.filters.push(v);
            }
            this.biases = new Vol(0, 0, 0, 0);
            this.biases.fromJSON(json.biases);
        }
    };

    let PoolLayer = function (opt) {
        opt = opt || {};

        this.sx = opt.sx;
        this.in_depth = opt.in_depth;
        this.in_sx = opt.in_sx;
        this.in_sy = opt.in_sy;

        this.sy = typeof opt.sy !== 'undefined' ? opt.sy : this.sx;
        this.stride = typeof opt.stride !== 'undefined' ? opt.stride : 2;
        this.pad = typeof opt.pad !== 'undefined' ? opt.pad : 0;

        this.out_depth = this.in_depth;
        this.out_sx = Math.floor((this.in_sx + this.pad * 2 - this.sx) / this.stride + 1);
        this.out_sy = Math.floor((this.in_sy + this.pad * 2 - this.sy) / this.stride + 1);
        this.layer_type = 'pool';

        this.switchx = zeros(this.out_sx * this.out_sy * this.out_depth);
        this.switchy = zeros(this.out_sx * this.out_sy * this.out_depth);
    };
    PoolLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let A = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
            let n = 0;
            for (let d = 0; d < this.out_depth; d++) {
                let x = -this.pad;
                let y = -this.pad;
                for (let ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                    y = -this.pad;
                    for (let ay = 0; ay < this.out_sy; y += this.stride, ay++) {
                        let a = -99999; // hopefully small enough ;\
                        let winx = -1, winy = -1;
                        for (let fx = 0; fx < this.sx; fx++) {
                            for (let fy = 0; fy < this.sy; fy++) {
                                let oy = y + fy;
                                let ox = x + fx;
                                if (oy >= 0 && oy < V.sy && ox >= 0 && ox < V.sx) {
                                    let v = V.get(ox, oy, d);
                                    if (v > a) {
                                        a = v;
                                        winx = ox;
                                        winy = oy;
                                    }
                                }
                            }
                        }
                        this.switchx[n] = winx;
                        this.switchy[n] = winy;
                        n++;
                        A.set(ax, ay, d, a);
                    }
                }
            }
            this.out_act = A;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            V.dw = zeros(V.w.length);
            let A = this.out_act;
            let n = 0;
            for (let d = 0; d < this.out_depth; d++) {
                let x = -this.pad;
                let y = -this.pad;
                for (let ax = 0; ax < this.out_sx; x += this.stride, ax++) {
                    y = -this.pad;
                    for (let ay = 0; ay < this.out_sy; y += this.stride, ay++) {
                        let chain_grad = this.out_act.get_grad(ax, ay, d);
                        V.add_grad(this.switchx[n], this.switchy[n], d, chain_grad);
                        n++;
                    }
                }
            }
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.sx = this.sx;
            json.sy = this.sy;
            json.stride = this.stride;
            json.in_depth = this.in_depth;
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            json.pad = this.pad;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
            this.sx = json.sx;
            this.sy = json.sy;
            this.stride = json.stride;
            this.in_depth = json.in_depth;
            this.pad = typeof json.pad !== 'undefined' ? json.pad : 0;
            this.switchx = zeros(this.out_sx * this.out_sy * this.out_depth);
            this.switchy = zeros(this.out_sx * this.out_sy * this.out_depth);
        }
    };

    let InputLayer = function (opt) {
        opt = opt || {};
        this.out_sx = typeof opt.out_sx !== 'undefined' ? opt.out_sx : opt.in_sx;
        this.out_sy = typeof opt.out_sy !== 'undefined' ? opt.out_sy : opt.in_sy;
        this.out_depth = typeof opt.out_depth !== 'undefined' ? opt.out_depth : opt.in_depth;
        this.layer_type = 'input';
    };
    InputLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            this.out_act = V;
            return this.out_act;
        },
        backward: function () {
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            var json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
        }
    };

    let SoftmaxLayer = function (opt) {
        opt = opt || {};
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = 'softmax';
    };
    SoftmaxLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let A = new Vol(1, 1, this.out_depth, 0.0);
            let as = V.w;
            let amax = V.w[0];
            for (let i = 1; i < this.out_depth; i++)
                if (as[i] > amax) amax = as[i];
            let es = zeros(this.out_depth);
            let esum = 0.0;
            for (let i = 0; i < this.out_depth; i++) {
                let e = Math.exp(as[i] - amax);
                esum += e;
                es[i] = e;
            }
            for (let i = 0; i < this.out_depth; i++) {
                es[i] /= esum;
                A.w[i] = es[i];
            }
            this.es = es;
            this.out_act = A;
            return this.out_act;
        },
        backward: function (y) {
            let x = this.in_act;
            x.dw = zeros(x.w.length); // zero out the gradient of input Vol

            for (let i = 0; i < this.out_depth; i++) {
                let indicator = i === y ? 1.0 : 0.0;
                let mul = -(indicator - this.es[i]);
                x.dw[i] = mul;
            }
            return -Math.log(this.es[y]);
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            json.num_inputs = this.num_inputs;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
            this.num_inputs = json.num_inputs;
        }
    };

    let RegressionLayer = function (opt) {
        opt = opt || {};
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = 'regression';
    };
    RegressionLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            this.out_act = V;
            return V;
        },
        backward: function (y) {
            let x = this.in_act;
            x.dw = zeros(x.w.length);
            let loss = 0.0;
            if (y instanceof Array || y instanceof Float64Array) {
                for (let i = 0; i < this.out_depth; i++) {
                    let dy = x.w[i] - y[i];
                    x.dw[i] = dy;
                    loss += 2 * dy * dy;
                }
            } else {
                let i = y.dim;
                let yi = y.val;
                let dy = x.w[i] - yi;
                x.dw[i] = dy;
                loss += 2 * dy * dy;
            }
            return loss;
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            json.num_inputs = this.num_inputs;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
            this.num_inputs = json.num_inputs;
        }
    };

    let SVMLayer = function (opt) {
        opt = opt || {};
        this.num_inputs = opt.in_sx * opt.in_sy * opt.in_depth;
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;
        this.layer_type = 'svm';
    };
    SVMLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            this.out_act = V;
            return V;
        },
        backward: function (y) {
            let x = this.in_act;
            x.dw = zeros(x.w.length);

            let yscore = x.w[y];
            let margin = 1.0;
            let loss = 0.0;
            for (let i = 0; i < this.out_depth; i++) {
                if (-yscore + x.w[i] + margin > 0) {
                    x.dw[i] += 1;
                    x.dw[y] -= 1;
                    loss += -yscore + x.w[i] + margin;
                }
            }
            return loss;
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            json.num_inputs = this.num_inputs;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
            this.num_inputs = json.num_inputs;
        }
    };

    let ReluLayer = function (opt) {
        opt = opt || {};
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.layer_type = 'relu';
    };
    ReluLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let V2 = V.clone();
            let N = V.w.length;
            let V2w = V2.w;
            for (let i = 0; i < N; i++)
                if (V2w[i] < 0) V2w[i] = 0;
            this.out_act = V2;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            let V2 = this.out_act;
            let N = V.w.length;
            V.dw = zeros(N);
            for (let i = 0; i < N; i++) {
                if (V2.w[i] <= 0) V.dw[i] = 0;
                else V.dw[i] = V2.dw[i];
            }
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
        }
    };

    let SigmoidLayer = function (opt) {
        opt = opt || {};
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.layer_type = 'sigmoid';
    };
    SigmoidLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let V2 = V.cloneAndZero();
            let N = V.w.length;
            let V2w = V2.w;
            let Vw = V.w;
            for (let i = 0; i < N; i++)
                V2w[i] = 1.0 / (1.0 + Math.exp(-Vw[i]));
            this.out_act = V2;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            let V2 = this.out_act;
            let N = V.w.length;
            V.dw = zeros(N);
            for (let i = 0; i < N; i++) {
                let v2wi = V2.w[i];
                V.dw[i] = v2wi * (1.0 - v2wi) * V2.dw[i];
            }
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
        }
    };

    let MaxoutLayer = function (opt) {
        opt = opt || {};
        this.group_size = typeof opt.group_size !== 'undefined' ? opt.group_size : 2;
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = Math.floor(opt.in_depth / this.group_size);
        this.layer_type = 'maxout';
        this.switches = global.zeros(this.out_sx * this.out_sy * this.out_depth); // useful for backprop
    };
    MaxoutLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let N = this.out_depth;
            let V2 = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
            if (this.out_sx === 1 && this.out_sy === 1) {
                for (let i = 0; i < N; i++) {
                    let ix = i * this.group_size;
                    let a = V.w[ix];
                    let ai = 0;
                    for (let j = 1; j < this.group_size; j++) {
                        let a2 = V.w[ix + j];
                        if (a2 > a) {
                            a = a2;
                            ai = j;
                        }
                    }
                    V2.w[i] = a;
                    this.switches[i] = ix + ai;
                }
            } else {
                let n = 0;
                for (let x = 0; x < V.sx; x++) {
                    for (let y = 0; y < V.sy; y++) {
                        for (let i = 0; i < N; i++) {
                            let ix = i * this.group_size;
                            let a = V.get(x, y, ix);
                            let ai = 0;
                            for (let j = 1; j < this.group_size; j++) {
                                let a2 = V.get(x, y, ix + j);
                                if (a2 > a) {
                                    a = a2;
                                    ai = j;
                                }
                            }
                            V2.set(x, y, i, a);
                            this.switches[n] = ix + ai;
                            n++;
                        }
                    }
                }

            }
            this.out_act = V2;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            let V2 = this.out_act;
            let N = this.out_depth;
            V.dw = global.zeros(V.w.length);
            if (this.out_sx === 1 && this.out_sy === 1) {
                for (let i = 0; i < N; i++) {
                    let chain_grad = V2.dw[i];
                    V.dw[this.switches[i]] = chain_grad;
                }
            } else {
                let n = 0;
                for (let x = 0; x < V2.sx; x++) {
                    for (let y = 0; y < V2.sy; y++) {
                        for (let i = 0; i < N; i++) {
                            let chain_grad = V2.get_grad(x, y, i);
                            V.set_grad(x, y, this.switches[n], chain_grad);
                            n++;
                        }
                    }
                }
            }
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            json.group_size = this.group_size;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
            this.group_size = json.group_size;
            this.switches = global.zeros(this.group_size);
        }
    };

    let tanh = (x) => {
        let y = Math.exp(2 * x);
        return (y - 1) / (y + 1);
    };

    let TanhLayer = function (opt) {
        opt = opt || {};
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.layer_type = 'tanh';
    };
    TanhLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let V2 = V.cloneAndZero();
            let N = V.w.length;
            for (let i = 0; i < N; i++) {
                V2.w[i] = tanh(V.w[i]);
            }
            this.out_act = V2;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            let V2 = this.out_act;
            let N = V.w.length;
            V.dw = zeros(N);
            for (let i = 0; i < N; i++) {
                let v2wi = V2.w[i];
                V.dw[i] = (1.0 - v2wi * v2wi) * V2.dw[i];
            }
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
        }
    }

    let DropoutLayer = function (opt) {
        opt = opt || {};
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.layer_type = 'dropout';
        this.drop_prob = typeof opt.drop_prob !== 'undefined' ? opt.drop_prob : 0.5;
        this.dropped = global.zeros(this.out_sx * this.out_sy * this.out_depth);
    };
    DropoutLayer.prototype = {
        forward: function (V, is_training) {
            this.in_act = V;
            if (typeof(is_training) === 'undefined') {
                is_training = false;
            }
            let V2 = V.clone();
            let N = V.w.length;
            if (is_training) {
                for (let i = 0; i < N; i++) {
                    if (Math.random() < this.drop_prob) {
                        V2.w[i] = 0;
                        this.dropped[i] = true;
                    } else {
                        this.dropped[i] = false;
                    }
                }
            } else {
                for (let i = 0; i < N; i++)
                    V2.w[i] *= this.drop_prob;
            }
            this.out_act = V2;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            let chain_grad = this.out_act;
            let N = V.w.length;
            V.dw = zeros(N);
            for (let i = 0; i < N; i++)
                if (!(this.dropped[i]))
                    V.dw[i] = chain_grad.dw[i];
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            json.drop_prob = this.drop_prob;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
            this.drop_prob = json.drop_prob;
        }
    };

    let LocalResponseNormalizationLayer = function (opt) {
        opt = opt || {};
        this.k = opt.k;
        this.n = opt.n;
        this.alpha = opt.alpha;
        this.beta = opt.beta;
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth;
        this.layer_type = 'lrn';
        if (this.n % 2 === 0)
            console.log('WARNING n should be odd for LRN layer');
    };
    LocalResponseNormalizationLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let A = V.cloneAndZero();
            this.S_cache_ = V.cloneAndZero();
            let n2 = Math.floor(this.n / 2);
            for (let x = 0; x < V.sx; x++) {
                for (let y = 0; y < V.sy; y++) {
                    for (let i = 0; i < V.depth; i++) {
                        let ai = V.get(x, y, i);
                        let den = 0.0;
                        for (let j = Math.max(0, i - n2); j <= Math.min(i + n2, V.depth - 1); j++) {
                            let aa = V.get(x, y, j);
                            den += aa * aa;
                        }
                        den *= this.alpha / this.n;
                        den += this.k;
                        this.S_cache_.set(x, y, i, den);
                        den = Math.pow(den, this.beta);
                        A.set(x, y, i, ai / den);
                    }
                }
            }
            this.out_act = A;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            V.dw = zeros(V.w.length);
            let A = this.out_act;
            let n2 = Math.floor(this.n / 2);
            for (let x = 0; x < V.sx; x++) {
                for (let y = 0; y < V.sy; y++) {
                    for (let i = 0; i < V.depth; i++) {
                        let chain_grad = this.out_act.get_grad(x, y, i);
                        let S = this.S_cache_.get(x, y, i);
                        let SB = Math.pow(S, this.beta);
                        let SB2 = SB * SB;
                        for (let j = Math.max(0, i - n2); j <= Math.min(i + n2, V.depth - 1); j++) {
                            let aj = V.get(x, y, j);
                            let g = -aj * this.beta * Math.pow(S, this.beta - 1) * this.alpha / this.n * 2 * aj;
                            if (j === i) g += SB;
                            g /= SB2;
                            g *= chain_grad;
                            V.add_grad(x, y, j, g);
                        }
                    }
                }
            }
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.k = this.k;
            json.n = this.n;
            json.alpha = this.alpha;
            json.beta = this.beta;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.out_depth = this.out_depth;
            json.layer_type = this.layer_type;
            return json;
        },
        fromJSON: function (json) {
            this.k = json.k;
            this.n = json.n;
            this.alpha = json.alpha;
            this.beta = json.beta;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.out_depth = json.out_depth;
            this.layer_type = json.layer_type;
        }
    };

    let QuadTransformLayer = function (opt) {
        opt = opt || {};
        this.out_sx = opt.in_sx;
        this.out_sy = opt.in_sy;
        this.out_depth = opt.in_depth + opt.in_depth * opt.in_depth;
        this.layer_type = 'quadtransform';
    };
    QuadTransformLayer.prototype = {
        forward: function (V) {
            this.in_act = V;
            let N = this.out_depth;
            let Ni = V.depth;
            let V2 = new Vol(this.out_sx, this.out_sy, this.out_depth, 0.0);
            for (let x = 0; x < V.sx; x++) {
                for (let y = 0; y < V.sy; y++) {
                    for (let i = 0; i < N; i++) {
                        if (i < Ni) {
                            V2.set(x, y, i, V.get(x, y, i));
                        } else {
                            let i0 = Math.floor((i - Ni) / Ni);
                            let i1 = (i - Ni) - i0 * Ni;
                            V2.set(x, y, i, V.get(x, y, i0) * V.get(x, y, i1));
                        }
                    }
                }
            }
            this.out_act = V2;
            return this.out_act;
        },
        backward: function () {
            let V = this.in_act;
            V.dw = zeros(V.w.length);
            let V2 = this.out_act;
            let N = this.out_depth;
            let Ni = V.depth;
            for (let x = 0; x < V.sx; x++) {
                for (let y = 0; y < V.sy; y++) {
                    for (let i = 0; i < N; i++) {
                        let chain_grad = V2.get_grad(x, y, i);
                        if (i < Ni) {
                            V.add_grad(x, y, i, chain_grad);
                        } else {
                            let i0 = Math.floor((i - Ni) / Ni);
                            let i1 = (i - Ni) - i0 * Ni;
                            V.add_grad(x, y, i0, V.get(x, y, i1) * chain_grad);
                            V.add_grad(x, y, i1, V.get(x, y, i0) * chain_grad);
                        }
                    }
                }
            }
        },
        getParamsAndGrads: function () {
            return [];
        },
        toJSON: function () {
            let json = {};
            json.out_depth = this.out_depth;
            json.out_sx = this.out_sx;
            json.out_sy = this.out_sy;
            json.layer_type = this.layer_type;
            return json;
        },
        fromJSON: function (json) {
            this.out_depth = json.out_depth;
            this.out_sx = json.out_sx;
            this.out_sy = json.out_sy;
            this.layer_type = json.layer_type;
        }
    };

    // set Layers
    app.ConvLayer = ConvLayer;
    app.FullyConnLayer = FullyConnLayer;
    app.PoolLayer = PoolLayer;
    app.InputLayer = InputLayer;
    app.RegressionLayer = RegressionLayer;
    app.SoftmaxLayer = SoftmaxLayer;
    app.SVMLayer = SVMLayer;
    app.TanhLayer = TanhLayer;
    app.MaxoutLayer = MaxoutLayer;
    app.ReluLayer = ReluLayer;
    app.SigmoidLayer = SigmoidLayer;
    app.DropoutLayer = DropoutLayer;
    app.LocalResponseNormalizationLayer = LocalResponseNormalizationLayer;
    app.QuadTransformLayer = QuadTransformLayer;

    // Class: Net
    let Net = function () {
        this.layers = [];
    };
    Net.prototype = {
        makeLayers: function (defs) {
            if (defs.length < 2)
                console.log('ERROR! For now at least have input and softmax layers.');
            if (defs[0].type !== 'input')
                console.log('ERROR! For now first layer should be input.');
            let desugar = function () {
                let new_defs = [];
                for (let i = 0; i < defs.length; i++) {
                    let def = defs[i];
                    if (def.type === 'softmax' || def.type === 'svm')
                        new_defs.push({type: 'fc', num_neurons: def.num_classes});
                    if (def.type === 'regression')
                        new_defs.push({type: 'fc', num_neurons: def.num_neurons});
                    if ((def.type === 'fc' || def.type === 'conv') && typeof(def.bias_pref) === 'undefined') {
                        def.bias_pref = 0.0;
                        if (typeof def.activation !== 'undefined' && def.activation === 'relu')
                            def.bias_pref = 0.1;
                    }
                    if (typeof def.tensor !== 'undefined')
                        if (def.tensor)
                            new_defs.push({type: 'quadtransform'});
                    new_defs.push(def);

                    if (typeof def.activation !== 'undefined') {
                        if (def.activation === 'relu')
                            new_defs.push({type: 'relu'});
                        else if (def.activation === 'sigmoid')
                            new_defs.push({type: 'sigmoid'});
                        else if (def.activation === 'tanh')
                            new_defs.push({type: 'tanh'});
                        else if (def.activation === 'maxout') {
                            let gs = def.group_size !== 'undefined' ? def.group_size : 2;
                            new_defs.push({type: 'maxout', group_size: gs});
                        }
                        else
                            console.log('ERROR unsupported activation ' + def.activation);
                    }

                    if (typeof def.drop_prob !== 'undefined' && def.type !== 'dropout')
                        new_defs.push({type: 'dropout', drop_prob: def.drop_prob});
                }
                return new_defs;
            };

            defs = desugar(defs);

            this.layers = [];
            for (let i = 0; i < defs.length; i++) {
                let def = defs[i];
                if (i > 0) {
                    let prev = this.layers[i - 1];
                    def.in_sx = prev.out_sx;
                    def.in_sy = prev.out_sy;
                    def.in_depth = prev.out_depth;
                }

                switch (def.type) {
                    case 'fc':
                        this.layers.push(new FullyConnLayer(def));
                        break;
                    case 'lrn':
                        this.layers.push(new LocalResponseNormalizationLayer(def));
                        break;
                    case 'dropout':
                        this.layers.push(new DropoutLayer(def));
                        break;
                    case 'input':
                        this.layers.push(new InputLayer(def));
                        break;
                    case 'softmax':
                        this.layers.push(new SoftmaxLayer(def));
                        break;
                    case 'regression':
                        this.layers.push(new RegressionLayer(def));
                        break;
                    case 'conv':
                        this.layers.push(new ConvLayer(def));
                        break;
                    case 'pool':
                        this.layers.push(new PoolLayer(def));
                        break;
                    case 'relu':
                        this.layers.push(new ReluLayer(def));
                        break;
                    case 'sigmoid':
                        this.layers.push(new SigmoidLayer(def));
                        break;
                    case 'tanh':
                        this.layers.push(new TanhLayer(def));
                        break;
                    case 'maxout':
                        this.layers.push(new MaxoutLayer(def));
                        break;
                    case 'quadtransform':
                        this.layers.push(new QuadTransformLayer(def));
                        break;
                    case 'svm':
                        this.layers.push(new SVMLayer(def));
                        break;
                    default:
                        console.log('ERROR: UNRECOGNIZED LAYER TYPE!');
                }
            }
        },

        // forward prop the network. A trainer will pass in is_training = true
        forward: function (V, is_training) {
            if (typeof(is_training) === 'undefined') is_training = false;
            var act = this.layers[0].forward(V, is_training);
            for (var i = 1; i < this.layers.length; i++) {
                act = this.layers[i].forward(act, is_training);
            }
            return act;
        },

        // backprop: compute gradients wrt all parameters
        backward: function (y) {
            var N = this.layers.length;
            var loss = this.layers[N - 1].backward(y); // last layer assumed softmax
            for (var i = N - 2; i >= 0; i--) { // first layer assumed input
                this.layers[i].backward();
            }
            return loss;
        },
        getParamsAndGrads: function () {
            // accumulate parameters and gradients for the entire network
            var response = [];
            for (var i = 0; i < this.layers.length; i++) {
                var layer_reponse = this.layers[i].getParamsAndGrads();
                for (var j = 0; j < layer_reponse.length; j++) {
                    response.push(layer_reponse[j]);
                }
            }
            return response;
        },
        getPrediction: function () {
            var S = this.layers[this.layers.length - 1]; // softmax layer
            var p = S.out_act.w;
            var maxv = p[0];
            var maxi = 0;
            for (var i = 1; i < p.length; i++) {
                if (p[i] > maxv) {
                    maxv = p[i];
                    maxi = i;
                }
            }
            return maxi;
        },
        toJSON: function () {
            var json = {};
            json.layers = [];
            for (var i = 0; i < this.layers.length; i++) {
                json.layers.push(this.layers[i].toJSON());
            }
            return json;
        },
        fromJSON: function (json) {
            this.layers = [];
            for (var i = 0; i < json.layers.length; i++) {
                var Lj = json.layers[i]
                var t = Lj.layer_type;
                var L;
                if (t === 'input') {
                    L = new InputLayer();
                }
                if (t === 'relu') {
                    L = new ReluLayer();
                }
                if (t === 'sigmoid') {
                    L = new SigmoidLayer();
                }
                if (t === 'tanh') {
                    L = new TanhLayer();
                }
                if (t === 'dropout') {
                    L = new DropoutLayer();
                }
                if (t === 'conv') {
                    L = new ConvLayer();
                }
                if (t === 'pool') {
                    L = new PoolLayer();
                }
                if (t === 'lrn') {
                    L = new LocalResponseNormalizationLayer();
                }
                if (t === 'softmax') {
                    L = new SoftmaxLayer();
                }
                if (t === 'regression') {
                    L = new RegressionLayer();
                }
                if (t === 'fc') {
                    L = new FullyConnLayer();
                }
                if (t === 'maxout') {
                    L = new MaxoutLayer();
                }
                if (t === 'quadtransform') {
                    L = new QuadTransformLayer();
                }
                if (t === 'svm') {
                    L = new SVMLayer();
                }
                L.fromJSON(Lj);
                this.layers.push(L);
            }
        }
    };

    app.Net = Net;

    // Class: Trainer
    let Trainer = function (net, options) {
        this.net = net;
        options = options || {};
        this.learning_rate = typeof options.learning_rate !== 'undefined' ? options.learning_rate : 0.01;
        this.l1_decay = typeof options.l1_decay !== 'undefined' ? options.l1_decay : 0.0;
        this.l2_decay = typeof options.l2_decay !== 'undefined' ? options.l2_decay : 0.0;
        this.batch_size = typeof options.batch_size !== 'undefined' ? options.batch_size : 1;
        this.method = typeof options.method !== 'undefined' ? options.method : 'sgd';

        this.momentum = typeof options.momentum !== 'undefined' ? options.momentum : 0.9;
        this.ro = typeof options.ro !== 'undefined' ? options.ro : 0.95;
        this.eps = typeof options.eps !== 'undefined' ? options.eps : 1e-6;

        this.k = 0; // iteration counter
        this.gsum = []; // last iteration gradients (used for momentum calculations)
        this.xsum = []; // used in adadelta
    };

    Trainer.prototype = {
        train: function (x, y) {
            let start = new Date().getTime();
            this.net.forward(x, true);
            let end = new Date().getTime();
            let fwd_time = end - start;

            start = new Date().getTime();
            let cost_loss = this.net.backward(y);
            let l2_decay_loss = 0.0;
            let l1_decay_loss = 0.0;
            end = new Date().getTime();
            let bwd_time = end - start;

            this.k++;
            if (this.k % this.batch_size === 0) {
                let pglist = this.net.getParamsAndGrads();
                if (this.gsum.length === 0 && (this.method !== 'sgd' || this.momentum > 0.0)) {
                    for (let i = 0; i < pglist.length; i++) {
                        this.gsum.push(zeros(pglist[i].params.length));
                        if (this.method === 'adadelta')
                            this.xsum.push(zeros(pglist[i].params.length));
                        else
                            this.xsum.push([]);
                    }
                }

                for (let i = 0; i < pglist.length; i++) {
                    let pg = pglist[i];
                    let p = pg.params;
                    let g = pg.grads;

                    let l2_decay_mul = typeof pg.l2_decay_mul !== 'undefined' ? pg.l2_decay_mul : 1.0;
                    let l1_decay_mul = typeof pg.l1_decay_mul !== 'undefined' ? pg.l1_decay_mul : 1.0;
                    let l2_decay = this.l2_decay * l2_decay_mul;
                    let l1_decay = this.l1_decay * l1_decay_mul;

                    let plen = p.length;
                    for (let j = 0; j < plen; j++) {
                        l2_decay_loss += l2_decay * p[j] * p[j] / 2; // accumulate weight decay loss
                        l1_decay_loss += l1_decay * Math.abs(p[j]);
                        let l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
                        let l2grad = l2_decay * (p[j]);
                        let gij = (l2grad + l1grad + g[j]) / this.batch_size; // raw batch gradient

                        let gsumi = this.gsum[i];
                        let xsumi = this.xsum[i];
                        if (this.method === 'adagrad') {
                            gsumi[j] = gsumi[j] + gij * gij;
                            let dx = -this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij;
                            p[j] += dx;
                        } else if (this.method === 'windowgrad') {
                            gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
                            let dx = -this.learning_rate / Math.sqrt(gsumi[j] + this.eps) * gij; // eps added for better conditioning
                            p[j] += dx;
                        } else if (this.method === 'adadelta') {
                            gsumi[j] = this.ro * gsumi[j] + (1 - this.ro) * gij * gij;
                            let dx = -Math.sqrt((xsumi[j] + this.eps) / (gsumi[j] + this.eps)) * gij;
                            xsumi[j] = this.ro * xsumi[j] + (1 - this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
                            p[j] += dx;
                        } else {
                            if (this.momentum > 0.0) {
                                let dx = this.momentum * gsumi[j] - this.learning_rate * gij; // step
                                gsumi[j] = dx; // back this up for next iteration of momentum
                                p[j] += dx; // apply corrected gradient
                            } else {
                                p[j] += -this.learning_rate * gij;
                            }
                        }
                        g[j] = 0.0; // zero out gradient so that we can begin accumulating anew
                    }
                }
            }

            return {
                fwd_time: fwd_time, bwd_time: bwd_time,
                l2_decay_loss: l2_decay_loss, l1_decay_loss: l1_decay_loss,
                cost_loss: cost_loss, softmax_loss: cost_loss,
                loss: cost_loss + l1_decay_loss + l2_decay_loss
            }
        }
    };

    app.Trainer = Trainer;
    app.SGDTrainer = Trainer;

    // Class: MagicNet
    let MagicNet = function (data, labels, opt) {
        opt = opt || {};
        if (typeof data === 'undefined')
            data = [];
        if (typeof labels === 'undefined')
            labels = [];

        this.data = data; // store these pointers to data
        this.labels = labels;

        this.train_ratio = getopt(opt, 'train_ratio', 0.7);
        this.num_folds = getopt(opt, 'num_folds', 10);
        this.num_candidates = getopt(opt, 'num_candidates', 50);
        this.num_epochs = getopt(opt, 'num_epochs', 50);
        this.ensemble_size = getopt(opt, 'ensemble_size', 10);
        this.batch_size_min = getopt(opt, 'batch_size_min', 10);
        this.batch_size_max = getopt(opt, 'batch_size_max', 300);
        this.l2_decay_min = getopt(opt, 'l2_decay_min', -4);
        this.l2_decay_max = getopt(opt, 'l2_decay_max', 2);
        this.learning_rate_min = getopt(opt, 'learning_rate_min', -4);
        this.learning_rate_max = getopt(opt, 'learning_rate_max', 0);
        this.momentum_min = getopt(opt, 'momentum_min', 0.9);
        this.momentum_max = getopt(opt, 'momentum_max', 0.9);
        this.neurons_min = getopt(opt, 'neurons_min', 5);
        this.neurons_max = getopt(opt, 'neurons_max', 30);

        this.folds = []; // data fold indices, gets filled by sampleFolds()
        this.candidates = []; // candidate networks that are being currently evaluated
        this.evaluated_candidates = []; // history of all candidates that were fully evaluated on all folds
        this.unique_labels = arrUnique(labels);
        this.iter = 0; // iteration counter, goes from 0 -> num_epochs * num_training_data
        this.foldix = 0; // index of active fold

        this.finish_fold_callback = null;
        this.finish_batch_callback = null;

        if (this.data.length > 0) {
            this.sampleFolds();
            this.sampleCandidates();
        }
    };
    MagicNet.prototype = {
        sampleFolds: function () {
            let N = this.data.length;
            let num_train = Math.floor(this.train_ratio * N);
            this.folds = []; // flush folds, if any
            for (let i = 0; i < this.num_folds; i++) {
                let p = randperm(N);
                this.folds.push({train_ix: p.slice(0, num_train), test_ix: p.slice(num_train, N)});
            }
        },
        sampleCandidate: function () {
            let input_depth = this.data[0].w.length;
            let num_classes = this.unique_labels.length;
            let layer_defs = [];
            layer_defs.push({type: 'input', out_sx: 1, out_sy: 1, out_depth: input_depth});
            let nl = weightedSample([0, 1, 2, 3], [0.2, 0.3, 0.3, 0.2]); // prefer nets with 1,2 hidden layers
            for (let q = 0; q < nl; q++) {
                let ni = randi(this.neurons_min, this.neurons_max);
                let act = ['tanh', 'maxout', 'relu'][randi(0, 3)];
                if (randf(0, 1) < 0.5) {
                    let dp = Math.random();
                    layer_defs.push({type: 'fc', num_neurons: ni, activation: act, drop_prob: dp});
                } else {
                    layer_defs.push({type: 'fc', num_neurons: ni, activation: act});
                }
            }
            layer_defs.push({type: 'softmax', num_classes: num_classes});
            let net = new Net();
            net.makeLayers(layer_defs);

            let bs = randi(this.batch_size_min, this.batch_size_max); // batch size
            let l2 = Math.pow(10, randf(this.l2_decay_min, this.l2_decay_max)); // l2 weight decay
            let lr = Math.pow(10, randf(this.learning_rate_min, this.learning_rate_max)); // learning rate
            let mom = randf(this.momentum_min, this.momentum_max); // momentum. Lets just use 0.9, works okay usually ;p
            let tp = randf(0, 1); // trainer type
            let trainer_def;
            if (tp < 0.33) {
                trainer_def = {method: 'adadelta', batch_size: bs, l2_decay: l2};
            } else if (tp < 0.66) {
                trainer_def = {method: 'adagrad', learning_rate: lr, batch_size: bs, l2_decay: l2};
            } else {
                trainer_def = {method: 'sgd', learning_rate: lr, momentum: mom, batch_size: bs, l2_decay: l2};
            }

            let trainer = new Trainer(net, trainer_def);

            let cand = {};
            cand.acc = [];
            cand.accv = 0; // this will maintained as sum(acc) for convenience
            cand.layer_defs = layer_defs;
            cand.trainer_def = trainer_def;
            cand.net = net;
            cand.trainer = trainer;
            return cand;
        },
        sampleCandidates: function () {
            this.candidates = []; // flush, if any
            for (let i = 0; i < this.num_candidates; i++) {
                let cand = this.sampleCandidate();
                this.candidates.push(cand);
            }
        },
        step: function () {
            this.iter++;
            let fold = this.folds[this.foldix]; // active fold
            let dataix = fold.train_ix[randi(0, fold.train_ix.length)];
            for (let k = 0; k < this.candidates.length; k++) {
                let x = this.data[dataix];
                let l = this.labels[dataix];
                this.candidates[k].trainer.train(x, l);
            }

            let lastiter = this.num_epochs * fold.train_ix.length;
            if (this.iter >= lastiter) {
                let val_acc = this.evalValErrors();
                for (let k = 0; k < this.candidates.length; k++) {
                    let c = this.candidates[k];
                    c.acc.push(val_acc[k]);
                    c.accv += val_acc[k];
                }
                this.iter = 0; // reset step number
                this.foldix++; // increment fold

                if (this.finish_fold_callback !== null) {
                    this.finish_fold_callback();
                }

                if (this.foldix >= this.folds.length) {
                    for (let k = 0; k < this.candidates.length; k++) {
                        this.evaluated_candidates.push(this.candidates[k]);
                    }
                    this.evaluated_candidates.sort(function (a, b) {
                        return (a.accv / a.acc.length) > (b.accv / b.acc.length) ? -1 : 1;
                    });
                    if (this.evaluated_candidates.length > 3 * this.ensemble_size) {
                        this.evaluated_candidates = this.evaluated_candidates.slice(0, 3 * this.ensemble_size);
                    }
                    if (this.finish_batch_callback !== null) {
                        this.finish_batch_callback();
                    }
                    this.sampleCandidates(); // begin with new candidates
                    this.foldix = 0; // reset this
                } else {
                    for (let k = 0; k < this.candidates.length; k++) {
                        let c = this.candidates[k];
                        let net = new Net();
                        net.makeLayers(c.layer_defs);
                        let trainer = new Trainer(net, c.trainer_def);
                        c.net = net;
                        c.trainer = trainer;
                    }
                }
            }
        },
        evalValErrors: function () {
            let vals = [];
            let fold = this.folds[this.foldix]; // active fold
            for (let k = 0; k < this.candidates.length; k++) {
                let net = this.candidates[k].net;
                let v = 0.0;
                for (let q = 0; q < fold.test_ix.length; q++) {
                    let x = this.data[fold.test_ix[q]];
                    let l = this.labels[fold.test_ix[q]];
                    net.forward(x);
                    let yhat = net.getPrediction();
                    v += (yhat === l ? 1.0 : 0.0); // 0 1 loss
                }
                v /= fold.test_ix.length; // normalize
                vals.push(v);
            }
            return vals;
        },
        predict_soft: function (data) {
            let nv = Math.min(this.ensemble_size, this.evaluated_candidates.length);
            if (nv === 0) {
                return new convnetjs.Vol(0, 0, 0);
            }
            let xout, n;
            for (let j = 0; j < nv; j++) {
                let net = this.evaluated_candidates[j].net;
                let x = net.forward(data);
                if (j === 0) {
                    xout = x;
                    n = x.w.length;
                } else {
                    for (let d = 0; d < n; d++) {
                        xout.w[d] += x.w[d];
                    }
                }
            }
            for (let d = 0; d < n; d++) {
                xout.w[d] /= n;
            }
            return xout;
        },
        predict: function (data) {
            let xout = this.predict_soft(data);
            let predicted_label = -1;
            if (xout.w.length !== 0) {
                let stats = maxmin(xout.w);
                predicted_label = stats.maxi;
            } else {
                predicted_label = -1; // error out
            }
            return predicted_label;
        },
        toJSON: function () {
            let nv = Math.min(this.ensemble_size, this.evaluated_candidates.length);
            let json = {};
            json.nets = [];
            for (let i = 0; i < nv; i++) {
                json.nets.push(this.evaluated_candidates[i].net.toJSON());
            }
            return json;
        },
        fromJSON: function (json) {
            this.ensemble_size = json.nets.length;
            this.evaluated_candidates = [];
            for (let i = 0; i < this.ensemble_size; i++) {
                let net = new Net();
                net.fromJSON(json.nets[i]);
                let dummy_candidate = {};
                dummy_candidate.net = net;
                this.evaluated_candidates.push(dummy_candidate);
            }
        },
        onFinishFold: function (f) {
            this.finish_fold_callback = f;
        },
        onFinishBatch: function (f) {
            this.finish_batch_callback = f;
        }
    };

    app.MagicNet = MagicNet;

    // Class: Functions
    let net = new Net();

    let _labels = {};
    let _labelIndex = 1;
    let _labelsReverse = {};

    let makeLayer = (layer) => {
        net.makeLayers(layer);
    };

    let setModel = (model) => {
        if (Array.isArray(model) || model.constructor === Object) {
            if (model.net) net.makeLayers(model.net);
            if (model.labelIndex) _labelIndex = model.labelIndex;
            if (model.labelsReverse) _labelsReverse = model.labelsReverse;
            if (model.labels) _labels = model.labels;
        } else if (fs.existsSync(model)) {
            let m = JSON.parse(fs.readFileSync(model, 'utf-8'));
            if (m.net) net.fromJSON(m.net);
            if (m.labelIndex) _labelIndex = m.labelIndex;
            if (m.labelsReverse) _labelsReverse = m.labelsReverse;
            if (m.labels) _labels = m.labels;
        }
    };

    let getModel = () => {
        return {net: net.toJSON(), labels: labels, labelsReverse: _labelsReverse, labelIndex: _labelIndex};
    };

    let trainOpts = {};
    let setOptions = (options) => trainOpts = options;

    let __trainer = null;

    let __checkVolume = (data) => {
        if (Array.isArray(data)) {
            data = new Vol(data);
        } else if (typeof data == 'object' && data.constructor !== 'CnnVolume') {
            let row = [];
            for (let key in data)
                row[key * 1] = data;
            data = new Vol(row);
        }
        return data;
    };

    let train = (dataset, labels) => {
        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];
        if (Array.isArray(labels) === false) labels = [labels];

        if (!trainOpts) trainOpts = {};
        if (!trainOpts.learning_rate) trainOpts.learning_rate = 0.01;
        if (!trainOpts.momentum) trainOpts.momentum = 0.9;
        if (!trainOpts.batch_size) trainOpts.batch_size = 2;
        if (!trainOpts.l2_decay) trainOpts.l2_decay = 0.01;

        for (let i = 0; i < labels.length; i++) {
            if (!_labels[labels[i]]) {
                _labels[labels[i]] = _labelIndex;
                _labelsReverse[_labelIndex] = labels[i];
                _labelIndex++;
            }
        }

        if (!__trainer) __trainer = new Trainer(net, trainOpts);
        for (let i = 0; i < dataset.length; i++)
            __trainer.train(__checkVolume(dataset[i]), _labels[labels[i]] - 1);
    };

    let test = (dataset, options) => {
        if (!options) options = {};
        if (Array.isArray(dataset) === false) dataset = [dataset];
        else if (typeof dataset[0] != 'object') dataset = [dataset];

        let result = [];
        for (let i = 0; i < dataset.length; i++) {
            let data = __checkVolume(dataset[i]);
            let avg = net.forward(data);
            let preds = [];
            for (let k = 0; k < avg.w.length; k++)
                preds.push({k: k, p: avg.w[k]});
            preds.sort(function (a, b) {
                return a.p < b.p ? 1 : -1;
            });

            let ans = _labelsReverse[preds[0].k + 1];
            if (options.score)
                result.push({answer: ans, score: preds});
            else
                result.push(ans);
        }

        if (result.length == 1) return result[0];
        else return result;
    };

    app.configure = setOptions;
    app.makeLayer = makeLayer;
    app.setModel = setModel;
    app.getModel = getModel;
    app.train = train;
    app.test = test;

    return app;
};