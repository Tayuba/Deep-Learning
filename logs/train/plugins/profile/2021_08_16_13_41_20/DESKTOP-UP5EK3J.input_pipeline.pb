	?A?????A????!?A????	?::^?E@?::^?E@!?::^?E@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:?A?????<I?fr??A?C?r????Y????e??rEagerKernelExecute 0*	?K7?A?Y@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat(??Z&á?!2dt???@@)??(?????1?G
?T?<@:Preprocessing2F
Iterator::Model? ?b??!?֔q?$F@)1[?*?1?????-<@:Preprocessing2U
Iterator::Model::ParallelMapV2???????!65?0@)???????165?0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap? ?K???!1a?z?/@)?N????1;?u!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice?h????~?!?!e?@)?h????~?1?!e?@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{????j??!)k??K@)??ฌ?z?1h9?2@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor?I?pt?!?z?r?@)?I?pt?1?z?r?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 32.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?::^?E@IU\???W@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?<I?fr???<I?fr??!?<I?fr??      ??!       "      ??!       *      ??!       2	?C?r?????C?r????!?C?r????:      ??!       B      ??!       J	????e??????e??!????e??R      ??!       Z	????e??????e??!????e??b      ??!       JCPU_ONLYY?::^?E@b qU\???W@