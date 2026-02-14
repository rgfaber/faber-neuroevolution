# Interoperability and Serialization

This guide covers how to export trained neural networks for deployment in production systems, including cross-language interoperability and serialization formats.

## Overview

After training with `faber_neuroevolution`, you'll have evolved networks that need to be deployed. Key considerations:

1. **Serialization** - Save/load networks for persistence
2. **Export formats** - ONNX, JSON, binary for cross-platform deployment
3. **Language interop** - Use trained networks in Python, Rust, C++, JavaScript
4. **Embedded systems** - Deploy to resource-constrained devices

## Serialization Options

### Erlang Term Format (Native)

The simplest option for Erlang/Elixir deployments:

```erlang
%% Save network to file
Network = Individual#individual.network,
ok = file:write_file("trained_network.etf", term_to_binary(Network)).

%% Load network from file
{ok, Binary} = file:read_file("trained_network.etf"),
Network = binary_to_term(Binary).

%% Use for inference
Inputs = [0.5, -0.3, 0.8, ...],
Outputs = network_evaluator:evaluate(Network, Inputs).
```

**Pros:** Fast, preserves all Erlang types, no conversion needed
**Cons:** Not portable to other languages

### JSON Export

For cross-language portability:

```erlang
%% Export network structure and weights to JSON
NetworkJson = faber_tweann:to_json(Network),
file:write_file("trained_network.json", NetworkJson).
```

JSON structure:

```json
{
  "topology": {
    "inputs": 42,
    "hidden_layers": [16, 8],
    "outputs": 6
  },
  "neurons": [
    {
      "id": 1,
      "type": "ltc",
      "bias": 0.234,
      "time_constant": 50.0,
      "state_bound": 1.0
    }
  ],
  "connections": [
    {
      "from": 1,
      "to": 10,
      "weight": 0.567
    }
  ]
}
```

### ONNX Export

For deployment with standard ML frameworks (PyTorch, TensorFlow, ONNX Runtime):

```erlang
%% Export to ONNX format
OnnxBinary = faber_tweann:to_onnx(Network),
file:write_file("trained_network.onnx", OnnxBinary).
```

Use in Python:

```python
import onnxruntime as ort

# Load the exported model
session = ort.InferenceSession("trained_network.onnx")

# Run inference
inputs = {"input": np.array([[0.5, -0.3, 0.8, ...]])}
outputs = session.run(None, inputs)
```

**Note:** ONNX export requires networks to use standard activation functions. LTC neurons are exported as approximations using RNN operators.

### Protocol Buffers

For high-performance RPC and streaming:

```erlang
%% Define in network.proto
%% message Network {
%%   repeated Neuron neurons = 1;
%%   repeated Connection connections = 2;
%% }

%% Export
ProtoBinary = network_to_protobuf(Network),
file:write_file("trained_network.pb", ProtoBinary).
```

## Cross-Language Deployment

### Python Integration

Using the JSON format:

```python
import json
import numpy as np

class EvolvedNetwork:
    def __init__(self, json_path):
        with open(json_path) as f:
            data = json.load(f)

        self.topology = data["topology"]
        self.neurons = {n["id"]: n for n in data["neurons"]}
        self.connections = data["connections"]
        self.weights = self._build_weight_matrix()

    def _build_weight_matrix(self):
        # Build weight matrices from connections
        ...

    def forward(self, inputs):
        # Feed-forward through network
        activations = inputs
        for layer_weights in self.weights:
            activations = np.tanh(np.dot(activations, layer_weights))
        return activations

# Usage
network = EvolvedNetwork("trained_network.json")
outputs = network.forward(sensor_data)
```

### Rust Integration

Using the JSON format with serde:

```rust
use serde::{Deserialize, Serialize};
use ndarray::Array1;

#[derive(Deserialize)]
struct Network {
    topology: Topology,
    neurons: Vec<Neuron>,
    connections: Vec<Connection>,
}

impl Network {
    pub fn from_json(path: &str) -> Self {
        let data = std::fs::read_to_string(path).unwrap();
        serde_json::from_str(&data).unwrap()
    }

    pub fn forward(&self, inputs: &Array1<f32>) -> Array1<f32> {
        // Implement forward pass
        ...
    }
}
```

### JavaScript/WebAssembly

For browser-based inference:

```javascript
class EvolvedNetwork {
  constructor(networkData) {
    this.topology = networkData.topology;
    this.weights = this.buildWeights(networkData);
  }

  forward(inputs) {
    let activations = inputs;
    for (const layerWeights of this.weights) {
      activations = this.tanh(this.matmul(activations, layerWeights));
    }
    return activations;
  }

  // For real-time applications, compile to WebAssembly
  static async loadWasm(wasmPath) {
    const module = await WebAssembly.instantiateStreaming(fetch(wasmPath));
    return module.instance.exports;
  }
}
```

## Embedded Deployment

### Nerves (Elixir on Embedded Linux)

For Raspberry Pi, BeagleBone, and similar devices:

```elixir
defmodule MyRobot.Brain do
  @network_path "/data/trained_network.etf"

  def start_link(opts) do
    GenServer.start_link(__MODULE__, opts, name: __MODULE__)
  end

  def init(_opts) do
    network = load_network(@network_path)
    {:ok, %{network: network}}
  end

  def infer(sensor_data) do
    GenServer.call(__MODULE__, {:infer, sensor_data})
  end

  def handle_call({:infer, inputs}, _from, %{network: network} = state) do
    outputs = :network_evaluator.evaluate(network, inputs)
    {:reply, outputs, state}
  end

  defp load_network(path) do
    path
    |> File.read!()
    |> :erlang.binary_to_term()
  end
end
```

### Microcontrollers (C/C++)

For Arduino, ESP32, STM32:

1. Export weights to C header:

```erlang
%% Generate C header with network weights
generate_c_header(Network, "network_weights.h").
```

Output:

```c
// network_weights.h - Auto-generated from evolved network
#ifndef NETWORK_WEIGHTS_H
#define NETWORK_WEIGHTS_H

#define INPUT_SIZE 42
#define HIDDEN1_SIZE 16
#define HIDDEN2_SIZE 8
#define OUTPUT_SIZE 6

const float weights_input_hidden1[INPUT_SIZE][HIDDEN1_SIZE] = {
  {0.234, -0.567, ...},
  ...
};

const float weights_hidden1_hidden2[HIDDEN1_SIZE][HIDDEN2_SIZE] = {
  ...
};

const float weights_hidden2_output[HIDDEN2_SIZE][OUTPUT_SIZE] = {
  ...
};

#endif
```

2. Use in embedded code:

```c
#include "network_weights.h"
#include <math.h>

float hidden1[HIDDEN1_SIZE];
float hidden2[HIDDEN2_SIZE];
float outputs[OUTPUT_SIZE];

void network_forward(float* inputs) {
    // Layer 1
    for (int j = 0; j < HIDDEN1_SIZE; j++) {
        float sum = 0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            sum += inputs[i] * weights_input_hidden1[i][j];
        }
        hidden1[j] = tanh(sum);
    }

    // Layer 2
    for (int j = 0; j < HIDDEN2_SIZE; j++) {
        float sum = 0;
        for (int i = 0; i < HIDDEN1_SIZE; i++) {
            sum += hidden1[i] * weights_hidden1_hidden2[i][j];
        }
        hidden2[j] = tanh(sum);
    }

    // Output layer
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        float sum = 0;
        for (int i = 0; i < HIDDEN2_SIZE; i++) {
            sum += hidden2[i] * weights_hidden2_output[i][j];
        }
        outputs[j] = tanh(sum);
    }
}
```

## Streaming Inference with Macula Mesh

For distributed inference across multiple nodes:

```erlang
%% On the inference node, register as an inference service
macula_rpc:register_method(<<"brain.infer">>, fun handle_inference/1).

handle_inference(#{<<"inputs">> := Inputs}) ->
    Outputs = network_evaluator:evaluate(Network, Inputs),
    #{<<"outputs">> => Outputs}.

%% On the robot/client, call the inference service
Outputs = macula_rpc:call(InferenceNode, <<"brain.infer">>, #{
    <<"inputs">> => SensorData
}).
```

## Best Practices

### Versioning

Always version your exported networks:

```erlang
export_network(Network, Version) ->
    #{
        version => Version,
        exported_at => calendar:universal_time(),
        network => Network,
        training_config => get_training_config(),
        fitness => Network#individual.fitness
    }.
```

### Validation

Validate exported networks before deployment:

```erlang
validate_export(Network, TestCases) ->
    lists:all(fun({Input, ExpectedOutput}) ->
        Output = network_evaluator:evaluate(Network, Input),
        max_error(Output, ExpectedOutput) < 0.01
    end, TestCases).
```

### Compression

For bandwidth-constrained deployments:

```erlang
%% Compress before sending over network
CompressedNetwork = zlib:compress(term_to_binary(Network)),

%% Decompress on receiving end
Network = binary_to_term(zlib:uncompress(CompressedNetwork)).
```

## Related Guides

- [Inference Scenarios](inference-scenarios.md) - Production deployment patterns
- [Swarm Robotics](swarm-robotics.md) - Distributed autonomous systems
- [Custom Evaluators](custom-evaluator.md) - Domain-specific training
