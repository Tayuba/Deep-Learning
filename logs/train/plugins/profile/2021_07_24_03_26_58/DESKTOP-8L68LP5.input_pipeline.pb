	sG?˵@sG?˵@!sG?˵@	???NG#@???NG#@!???NG#@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:sG?˵@?^Cp\ƽ?A{?????@Y???????rEagerKernelExecute 0*	?S㥛:s@2F
Iterator::Model2??????!u????R@)???(_???1OX2??Q@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapr5?+-#??!???O?*@)?D???V??1ӕz<?&@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?:␝?!ߎ8???"@)Ɖ?v???1???X"?@:Preprocessing2U
Iterator::Model::ParallelMapV2P??????!]?C?@)P??????1]?C?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice????bc~?!+?H??J@)????bc~?1+?H??J@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip+???ڧ??!-?[??8@)?*l? {?1t?sP?8@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor=E7?r?!
???ܮ??)=E7?r?1
???ܮ??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 9.6% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s4.2 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9???NG#@I.?|!?V@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?^Cp\ƽ??^Cp\ƽ?!?^Cp\ƽ?      ??!       "      ??!       *      ??!       2	{?????@{?????@!{?????@:      ??!       B      ??!       J	??????????????!???????R      ??!       Z	??????????????!???????b      ??!       JCPU_ONLYY???NG#@b q.?|!?V@