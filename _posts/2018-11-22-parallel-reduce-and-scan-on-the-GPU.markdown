---
layout: post
title:  "Parallel reduce and scan on the GPU"
categories: vulkan
---

### Introduction

GPUs are formidable parallel machines, capable of running thousands of threads simultaniously. They are excellent for embarassily parallel algorithms. There are many APIs to interact and use them: OpenGL, DirectX, OpenCL, CUDA and Vulkan. They each their pros and cons, and different usages. In particular Vulkan is a modern API and multi-platform, and with version 1.1 has added new interesting capabilities, in particular subgroups. 

To understand what a subgroup is, let's review the abstract model used by Vulkan. Tasks are divided in a number of work groups, themselves in a number of subgroups, and each one is divided in a number of invocation, corresponding to individual tasks

![GPU abstract model](/blog/assets/gpu.png)

Each work group has its own cache that can be accessed directly by the tasks, called shared memory. Obviously accessing this memory is much faster than the global memory and algorithms are designed to make as much us of it as possible.
The next subdivision, workgroups, are essentially large SIMD groups that execute the tasks in lockstep. Those SIMD can communicate with special instructins, bypassing the shared memory or global memory. They can be used to accelerate the tasks.

Let's have a look at two parallel algorithms, and how to implement them with subgroups: reduce and scan.

### Vulkan subgroups

Vulkan subgroups are available with version 1.1, we can query the capabilitiese provided by the GPU to get some information on the size of subgroups and which operations are supported

{% highlight cpp %}
auto properties = 
  physicalDevice.getProperties2<vk::PhysicalDeviceProperties2, vk::PhysicalDeviceSubgroupProperties>();
auto subgroupProperties = 
  properties.get<vk::PhysicalDeviceSubgroupProperties>();

std::cout << "Subgroup size: " 
          << subgroupProperties.subgroupSize 
          << std::endl;

std::cout << "Subgroup supported operations: " 
          << vk::to_string(subgroupProperties.supportedOperations) 
          << std::endl;
{% endhighlight %}

On my machine with a Vega 56, the following is returned:

    Subgroup size: 64
    Subgroup supported operations: {Basic | Vote | Arithmetic | Ballot | Shuffle | ShuffleRelative | Quad}

Arithmetic is the type of operation we'll need to implement scan and reduce, and introduction to the other operations can be found at this [khronos vulkan subgroup](https://www.khronos.org/blog/vulkan-subgroup-tutorial)

### Reduce

Reduce is very simple, it takes a list of elements {% latex %} x_0, x_1, x_2, ... {% endlatex %} and calculates its sum,

{% latex centred %} 
x = \sum_{i=0}^n x_i
{% endlatex %}

C++17 has added it as `std::reduce` which can be run in parallel or sequentially.
The equivalent operation in Vulkan for subgroups is:

{% highlight glsl %}
float sum = subgroupAdd(value);
{% endhighlight %}

Every invocation belonging to the subgroup will return the total sum.

![Reduce](/blog/assets/reduce.png){:width="75%"}

We can reduce up to 64 values on my machine. To extend this to bigger sizes, we need to save the sum of each subgroup in a workgroup in the shared memory. Assuming we have less number of subgroups then the size of a subgroup, we load those values from the shared memory in the first subgroup and call `subgroupAdd` again. 

![Subgroupp reduce](/blog/assets/subgroup_reduce.png)

To reduce a bigger number of elements than a workgroup can hold, we need to run the algorithm again. At the end of the first pass, we'll have `N` number of elements corresponding to `N` workgroups, since we can't synchronize (without using atomic) between workgroups, we need to insert a memory barrier and invoke again `N` invocations which can be reduced to one element.

{% highlight glsl %}
shared float sdata[sumSubGroupSize];

void main()
{
    float sum = 0.0;
    if (gl_GlobalInvocationID.x < consts.n)
    {
        sum = inputs[gl_GlobalInvocationID.x];
    }

    sum = subgroupAdd(sum);

    if (gl_SubgroupInvocationID == 0)
    {
        sdata[gl_SubgroupID] = sum;
    }

    memoryBarrierShared();
    barrier();

    if (gl_SubgroupID == 0)
    {
        sum = gl_SubgroupInvocationID < gl_NumSubgroups ? 
          sdata[gl_SubgroupInvocationID] : 0;
        sum = subgroupAdd(sum);
    }

    if (gl_LocalInvocationID.x == 0)
    {
        outputs[gl_WorkGroupID.x] = sum;
    }
}
{% endhighlight %}

Let's see how fast this algorithm is, comparing it with `std::reduce` running in sequence and in parallel. We're also comparing with a regulard reduce algorithm using only shared memory, based on the excellent slides from Mark Harris: [Optimizing parallel reduce in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

<figure>
  <embed type="image/svg+xml" src="/blog/assets/reduce.svg" />
</figure>

That's rather disapointing, the subgroup based reduce is only slightly faster. However it is much easier to implement than the shared memory based one.

# Scan

Scan, or prefix sum takes a list of elements {% latex %} x_0, x_1, x_2, ... {% endlatex %} and produces a sequence of numbers {% latex %} y_0, y_1, y_2, ... {% endlatex %} such that,

{% latex centred %}
\begin{aligned}
y_0 &= x_0 \\
y_1 &= x_0 + x_1 \\
y_2 &= x_0 + x_1 + x_2 \\
&...
\end{aligned}
{% endlatex %}

Again, this is available in C++17 with `std::inclusive_scan`. The vulkan operation is,

{% highlight glsl %}
float value = subgroupInclusiveAdd(value);
{% endhighlight %}

Similarily to reduce, each invocation in the subgroup will receive the partial sum corresponding to its index (in increasing order).
![Scan](/blog/assets/scan.png){:width="60%"}

We will use a similar strategy as for reduce to be able to scan over a bigger number of elements than the subgroup size.

![Subgorup scan](/blog/assets/subgroup_scan.png)

Again, running a scan on the CPU, GPU with shared memory and GPU with subgroup operations gives the following:

<figure>
  <embed type="image/svg+xml" src="/blog/assets/scan.svg" />
</figure>

This is very impressive, much better improvements than with the reduce with subgroups!