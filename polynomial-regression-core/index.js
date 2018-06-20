
import * as tf from '@tensorflow/tfjs';
const a1 = tf.variable(tf.scalar(Math.random()))
const b1 = tf.variable(tf.scalar(Math.random()))


function predict(x){
  //ax+b
  return tf.tidy(() => {
    return a1.mul(x).add(b1);
  })
}

function loss(predictions, labels){
  const meanSquareError = predictions.sub(labels).square().mean();
  return meanSquareError;
}


function train(xs, ys, numIterations = 75) {

  const learningRate = 0.05;
  const optimizer = tf.train.sgd(learningRate);

  for (let iter = 0; iter < numIterations; iter++) {
    // console.log("count : "+iter,a1.print(),b1.print());
    optimizer.minimize(() => {
      const predsYs = predict(xs);
      // console.log("pys:"+predsYs.print()+" and xs:"+ xs.print());
      return loss(predsYs, ys);
    });
  }
}

var arry=[],arrx=[];
for(var i =0.0;i<2;i+=0.01){
var a=25,b=5;
var y = a*i + b;
arrx.push(i);
arry.push(y);
}




const xs = tf.tensor1d(arrx);
const ys = tf.tensor1d(arry);

const pYs = predict(xs);
// console.log(pYs.print(),ys.print());
// console.log(loss(pYs, ys).print());

train(xs,ys,1000);

console.log(a1.print(),b1.print());
