class ResNN {
    constructor(){

        //CONTROL VARIABLES
        //Input vector
        this.input = [.9,.7,.1,0,0,.1,0,.36];

        this.inputSize = 50;

        this.rcOn = true;

        //Vector the AI is training to match
        this.targetVector = [.5,0,0,(0-.5),0,0,.3,0];

        //Hyperparameters
        this.learningFactor = .1;
        this.percentUpdated = .008;
        this.bias = .0001;
        this.reservoirPercentage = .1;
        this.reservoirGearRatio = 4;

        //Loop iterations
        this.runs = 1000;
        ////////////////////////

        //MAIN VARIABLES
        this.inputVector = [];

        this.errorVectorL1 = [];
        this.errorVectorL2 = [];

        this.errorGradientL2 = [];

        this.outputVector1 = [];
        this.outputVector2 = [];

        this.outputVectorA1 = [];
        this.outputVectorA2 = [];

        this.dOutput1 = [];

        this.transformedVector = [];
        this.transformedVectorTemp = [];

        this.gradientMatrix2 = this.createZMatrix(this.inputSize);
        this.gradientMatrix1 = this.createZMatrix(this.inputSize);
        //END MAIN VARIABLES

        this.inputVector = this.input;

        //PAD INPUT AND TARGET VECTORS
        for(let i = this.input.length; i < this.inputSize; i++){
            this.inputVector[i] = 0;
        }

        for(let i = this.targetVector.length; i < this.inputSize; i++){
            this.targetVector[i] = 0;
        }

        this.dErrorVectr2 = [];
        this.dOutput2 = [];

        this.weightMatrix1 = [];
        this.weightMatrix2 = [];
        this.weightMatrix2Transpose = [];

        this.reservoirMatrix = [];
        this.normalizedProjectedInput = [];
    }

//END CONSTRUCTOR
//END PADDING

//MATRIX GENERATION UTIL FUNCTIONS
//Matrix generator that randomizes all cells
create0Matrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let k = 0; k < diameter; k++) {
            matrix[i][k] = Math.random() - .5;
        }
      }
    return matrix;
}

//Matrix generator that zeroes all cells
createZMatrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let k = 0; k < diameter; k++) {
            matrix[i][k] = 0;
        }
      }
    return matrix;
}
//END MATRIX GENERATION UTIL FUNCTIONS

//MISC FUNCTIONS
dActivation(x){
    return 4/(Math.pow((Math.exp((0-1)*x)+Math.exp(x)),2));
}
//END MISC FUNCTIONS

createResMatrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let j = 0; j < diameter; j++) {
            if(Math.random() < this.reservoirPercentage){
                matrix[i][j] = Math.random() - .5;
            }else{
                matrix[i][j] = 0;
            }
        }
      }
    return matrix;
}

//BEGIN FORWARD PASS
//Matrix Multiplies Input Vector by Weight Matrix and Applies Standard Tanh Activation Function
matrixMultiply1(inputV, weightM, diam){
    var outputVector = [];
    for(let i = 0; i < diam; i++){
        var sum = 0;
        for(let k = 0; k < diam; k++){
            sum = sum + (inputV[k] * weightM[i][k]);
        }
        this.outputVector1[i] = sum;
        this.outputVectorA1[i] = Math.tanh(sum + this.bias);
        outputVector[i] = Math.tanh(sum + this.bias);
    }
    return outputVector;
}

matrixMultiply2(inputV, weightM, diam){
    var outputVector = [];
    for(let i = 0; i < diam; i++){
        var sum = 0;
        for(let k = 0; k < diam; k++){
            sum = sum + (inputV[k] * weightM[i][k]);
        }
        this.outputVector2[i] = sum;
        this.outputVectorA2[i] = Math.tanh(sum + this.bias);
        outputVector[i] = Math.tanh(sum + this.bias);
    }
    return outputVector;
}


//END FORWARD PASS

//BACKWARD PASS UTIL FUNCTIONS


//RESERVOIR

//Reservoir computer preprocesses temporally correlated inputs
createResMatrix(diameter){
    var matrix = [];

    for (let i = 0; i < diameter; i++) {
        matrix[i] = [];
        for(let k = 0; k < diameter; k++) {
            if(Math.random() < this.reservoirPercentage){
                matrix[i][k] = Math.random() - .5;
            }else{
                matrix[i][k] = 0;
            }
        }
      }
    return matrix;
}

updateReservoirProjection(inputV, weightM, diam, zeroIndex, initialInputProjection){
    var outputVector = [];
    for(let i = 0; i < diam; i++){
        if(i >= zeroIndex){
            inputV[i] = initialInputProjection[i] + inputV[i];
        }
    }
    for(let i = 0; i < diam; i++){
        var sum = 0;
        for(let j = 0; j < diam; j++){
            sum = sum + (inputV[j] * weightM[i][j]);
        }
        outputVector[i] = Math.tanh(sum + this.bias);
    }
    initialInputProjection = outputVector;
    return outputVector;
}

normalizeInputs(projectedInputs){
    var sum = 0;
    for(let i = 0; i < projectedInputs.length; i++){
        sum = sum + projectedInputs[i];
    }
    for(let i = 0; i < projectedInputs.length; i++){
        projectedInputs[i] = projectedInputs[i] / sum;
    }
    return projectedInputs;
}                                                                                                            
//RESERVOIR

backPropagateO(arg1, arg2, arg3, arg4){
    let dError = [];
    let bpWeights = [];
    let outV = []
    let dOutput = [];
    bpWeights = arg1;
    dError = arg2;
    outV = arg3;
    dOutput = arg4;

    
    for(let i = 0; i < this.inputSize; i++){
        for(let j = 0; j < this.inputSize; j++){
            this.gradientMatrix2[i][j] = (this.learningFactor*this.outputVector1[j]*dOutput[i][j]*dError[i][j]);
        }
    }

    for(let i = 0; i < this.inputSize; i++){
        for(let j = 0; j < this.inputSize; j++){
            bpWeights[i][j] = bpWeights[i][j] + this.gradientMatrix2[i][j];
        }
    }

    return bpWeights;
}



backPropagateH1(arg1, arg3, arg4, arg5){
    let dErrorMost = this.createZMatrix(this.inputSize);
    let bpWeights = [];
    let outV = []
    let dOutput = [];
    let outputA = [];
    let E = 0;
    bpWeights = arg1;
    outV = arg3;
    dOutput = arg4;
    outputA = arg5;


    //Calculate dErrorMost//
    for(let i = 0; i < this.inputSize; i++){
        E = 0;
        for(let j = 0; j < this.inputSize; j++){
                E = E + this.weightMatrix2Transpose[i][j]*this.dErrorVectr2[i][j]*this.dOutput2[i][j];
            dErrorMost[i][j] = E;
        }
    }

    for(let i = 0; i < this.inputSize; i++){
        for(let j = 0; j < this.inputSize; j++){
            this.gradientMatrix1[i][j] = (this.learningFactor*this.normalizedProjectedInput[i]*dOutput[j]*dErrorMost[i][j]);
        }
    }

    for(let i = 0; i < this.inputSize; i++){
        for(let j = 0; j < this.inputSize; j++){
            bpWeights[i][j] = bpWeights[i][j] + this.gradientMatrix1[i][j];
        }
    }

    return bpWeights;
}



//END BACKWARD PASS UTIL FUNCTIONS
train(){

this.dErrorVectr2 = this.createZMatrix(this.inputSize);
this.dOutput2 = this.createZMatrix(this.inputSize);

this.weightMatrix1 = this.create0Matrix(this.inputSize);
this.weightMatrix2 = this.create0Matrix(this.inputSize);
this.weightMatrix2Transpose = this.create0Matrix(this.inputSize);

this.reservoirMatrix = this.createResMatrix(this.inputSize);

//New randomized square weight matrix
let projectedInputVector = [];
for(let i = 0; i < this.inputSize; i++){
    projectedInputVector[i] = 0;
}



for(let r = 0; r < this.runs; r++){
    var errorInitial = 0;
    // var tempError = 0;
    // var finalError = 0;

    if(this.rcOn == true){
        projectedInputVector = this.updateReservoirProjection(this.inputVector, this.reservoirMatrix, this.inputSize, this.input.length, projectedInputVector);
        this.normalizedProjectedInput = this.normalizeInputs(projectedInputVector);
        this.reservoirGearRatio = 1;
    } else{
        this.normalizedProjectedInput = this.inputVector;
    }

    for(let a = 0; a < this.reservoirGearRatio; a++){

        this.transformedVector = this.matrixMultiply2(this.matrixMultiply1(this.normalizedProjectedInput, this.weightMatrix1, this.inputVector.length), this.weightMatrix2, this.inputVector.length);
        //console.log(transformedVector);

        for(let i = 0; i < this.inputSize; i++){
            this.dOutput1[i] = this.dActivation(this.outputVector1[i]);
        }

    
        for(let i = 0; i < this.inputSize; i++){
            for(let j = 0; j < this.inputSize; j++){
                this.dOutput2[i][j] = this.dActivation(this.outputVector2[i]);
            }
        }

    
        for(let k = 0; k < this.inputSize; k++){
            this.errorVectorL2[k] = ((Math.pow((this.targetVector[k] - this.transformedVector[k]),2))/2);
        }


        for(let i = 0; i < this.inputSize; i++){
            for(let j = 0; j < this.inputSize; j++){
                this.dErrorVectr2[i][j] = (this.targetVector[i] - this.transformedVector[i]);
            }
        }


        for(let k = 0; k < this.inputSize; k++){
            errorInitial = errorInitial + ((Math.pow((this.targetVector[k] - this.transformedVector[k]),2))/2);
        }

        //Backpropogate Weight Matrix 2
        this.weightMatrix2 = this.backPropagateO(this.weightMatrix2, this.dErrorVectr2, this.outputVectorA1, this.dOutput2);

        //Transpose Weight Matrix 2
        for(let s = 0; s < this.inputSize; s++){
            for(let t = 0; t < this.inputSize; t++){
                this.weightMatrix2Transpose[t][s] = this.weightMatrix2[s][t];
            }
        }

        //Backpropogate Weight Matrix 1
        this.weightMatrix1 = this.backPropagateH1(this.weightMatrix1, this.outputVector1, this.dOutput1, this.outputVectorA1);

        console.log("Error: "+errorInitial.toExponential());
    }
}
}

}

const nn1 = new ResNN();
nn1.train();