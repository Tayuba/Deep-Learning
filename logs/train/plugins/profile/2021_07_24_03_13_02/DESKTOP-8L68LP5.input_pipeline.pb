	W?/?'?@W?/?'?@!W?/?'?@	?Yڴ?C@?Yڴ?C@!?Yڴ?C@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:W?/?'?@?JY?8???A???<,T @YtF??_ @rEagerKernelExecute 0*	?????t?@2F
Iterator::Model??JY?8??!?5? ?U@)jM????1?:???T@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?ݓ??Z??!m{&t@)7?[ A??1a???@:Preprocessing2U
Iterator::Model::ParallelMapV2Q?|a2??!??ȣBL	@)Q?|a2??1??ȣBL	@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????!JW?D??@)?Q?????1???kc@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?o_???!	?|?h@)?o_???1	?|?h@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip䃞ͪϵ?!? T^?*@)F%u???1T^??! @:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor	?^)?p?!Y7@?
??)	?^)?p?1Y7@?
??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 39.3% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*high2t21.4 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?Yڴ?C@IZ?%K?TN@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?JY?8????JY?8???!?JY?8???      ??!       "      ??!       *      ??!       2	???<,T @???<,T @!???<,T @:      ??!       B      ??!       J	tF??_ @tF??_ @!tF??_ @R      ??!       Z	tF??_ @tF??_ @!tF??_ @b      ??!       JCPU_ONLYY?Yڴ?C@b qZ?%K?TN@