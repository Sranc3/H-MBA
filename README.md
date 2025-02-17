# H-MBA
H-MBA: Hierarchical MamBa Adaptation for Multi-Modal Video Understanding in Autonomous Driving （AAAI 2025）
### Introduction for H-MBA ###

With the prevalence of Multimodal Large Language Models(MLLMs), autonomous driving has encountered new opportunities and challenges. 
In particular, multi-modal video understanding is critical to interactively analyze what will happen in the procedure of autonomous driving.
However, videos in such a dynamical scene that often contains complex spatial-temporal movements,
which restricts the generalization capacity of the existing MLLMs in this field.
To bridge the gap, 
we propose a novel $\textbf{H}ierarchical$ $\textbf{M}am\textbf{ba}$ $Adaptation$ (H-MBA) framework to fit the complicated motion changes in autonomous driving videos.
It has two main structures, e.g., C-Mamba and Q-Mamba.
C-Mamba contains various types of structure state space models,
which can effectively capture multi-granularity video context for different temporal resolutions.
Q-Mamba flexibly transforms the current frame as the learnable query, 
and attentively selects multi-granularity video context into query.
Consequently,
it can adaptively integrate all the video contexts of multi-scale temporal resolutions to enhance video understanding.
Via a plug-and-play paradigm in MLLMs,
our H-MBA enables image based MLLM models to understand video contents, 
and shows remarkable performance on multi-modal video tasks in autonomous driving,
e.g., for risk object detection, 
it outperforms the previous SOTA method with 5.5\% mIoU improvement.


<div  align="center">    
	<img src="/images/frame.png" width="80%" alt="Figure 1">
	<p>Figure 1：Overview of the framewrok</p>
</div>

<div  align="center">    
	<img src="images/threeMB.jpg" width="80%" alt="Figure 2:Illustration of three different mamba modules">
	<p>Figure 2: Illustration of three different mamba modules</p>
</div>




## Application and Limitation
Traditional models are often limited to predefined questions, limiting their applicaiton in open-world situations.
Benefiting from the powerful reasoning capabilities of LLM models, our method exhibits good generalization ability, 
enabling it to be directly applied to real-world scenarios for simple question-answering conversationals in a unified paradigm. 
For example, it can effectively provide driving risk warning alerts as shown in the image below.
However, the model's responses remain highly dependent on the examples in the training data. 
In real-world application scenarios, we often encounter long-tail cases that are not included in the training set, 
such as suddenly dropped cargo from a vehicle ahead, road obstacles, or animals unexpectedly crossing the path. 
In such situations, the model often fail to make correct judgments.
<div  align="center">    
	<img src="images/real.png" width="80%" alt="Figure 3: Real world applications">
	<p>Figure 3: Real world applications</p>
</div>
And we have explored to solve such problem in our next paper.



## Data Preparation ##
DRAMA explores multiple facets of joint risk localization and captioning in interactive driving scenarios. In particular, they benchmark various multi-task prediction architectures and provide a detailed analysis of joint risk localization and risk captioning. The data set is available at https://usa.honda-ri.com/drama. They provide many frame images of a video and optical flow features, we only use 5 frames for temporal modeling.

BDD-X Dataset coule be available at https://github.com/JinkyuKimUCB/BDD-X-dataset for detailed information, and we cut the frames from the raw videos.


## Acknowledgement

This repo benefits from [LLaVA](https://github.com/haotian-liu/LLaVA), [Vicuna](https://github.com/lm-sys/FastChat), [ChatGLM-Efficient-Tuning](https://github.com/hiyouga/ChatGLM-Efficient-Tuning) and [GLIGEN](https://github.com/gligen/GLIGEN), [Shikra](https://github.com/shikras/shikra). Thanks for their wonderful works.
