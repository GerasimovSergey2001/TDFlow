# Temporal Difference Flows

This repository provides the code for [Conditional Flow Matching](https://arxiv.org/abs/2210.02747) and [Temporal Difference Flows](https://arxiv.org/pdf/2503.09817).

### Experiments

The experiments were performed for the following tasks in the **PointMass Maze** environment (from [dm_control](https://github.com/google-deepmind/dm_control)):

* **Reach Top Left**: The agent must navigate to the upper-left corner of the maze.
* **Reach Top Right**: The agent must navigate to the upper-right corner of the maze.

These tasks evaluate the ability of [Temporal Difference Flows](https://arxiv.org/pdf/2503.09817) to capture the discounted occupancy distribution (Successor Measure).

## Installation

```bash
uv sync
```

## Data Setup
 
```bash
download_exorl.sh
```

## Usage

### 1. Expetr Policy Training

To train the agent with expert policies run `demo_td3.ipynb` for tasks reach_top_right and reach_top_left.

This gives `td3_point_mass_expert_{task}.zip` and `agent_trajectory_{task}.gif`.

This is necessary to reproduce the results from the article.


### 2. Run Training

To launch the training process, run the following command in your terminal:

```bash
python3 -m train --task reach_top_left --num_epochs 100 --loss_type td2_cfm
```
Arguments:

--task: The target environment task. Either reach_top_left (default) or reach_top_right

--num_epochs: Number of training epochs (integer). Default: 100.

--loss_type: The objective function used for training: either td_cfm or td2_cfm.

For PointMass Maze tasks, we recommend at least 500 epochs to achieve high-fidelity Successor Measure approximations as described in [the original article](https://arxiv.org/pdf/2503.09817).

This generates 

- `checkpoints/{loss_type}_model_{task}_epoch_{epoch}.pth`

- `checkpoints/{loss_type}_model_{task}.pth`

### 3. Alternative Training 

Run `demo_tdflow.ipynb` providing necessary configuration in Google Colab .

### 4. Evaluation

To launch evaluation, run the following command in your terminal:

```bash
python3 -m evaluate --task reach_top_left --model td2_cfm --epoch None
```
Arguments:

--task: The target environment task. Either reach_top_left (default) or reach_top_right.

--model: model obtained from the objective function used for training: either td_cfm or td2_cfm (default).

--epoch: uploading model from a given checkpoint (checkpoints are provided for multiples of 5 epochs). The final model can be obtained setting epoch to None.

This generates evaluation metrics (with standard deviations) for a task.

## Additional Notebooks

To demonstarte that Conditional Flow Matching is implemented correctly, we provide conditioned 2D guassian mixtures  example in `conditional_flow_matching.ipynb`.

## Model Storage

Models' weights are stored in https://huggingface.co/SergeiGerasimov/TDFlow

Models' losses can be found at https://wandb.ai/gerasimov-serf/TDFlow-Project/table?nw=nwusergerasimovserf
