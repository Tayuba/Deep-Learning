	k(??v??k(??v??!k(??v??	???? 	@???? 	@!???? 	@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:k(??v????S????A7?C$??Y??y??w??rEagerKernelExecute 0*	?K7?A?R@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?q?PiĜ?!?NI?B@)-AF@?#??13???@{?@:Preprocessing2U
Iterator::Model::ParallelMapV2???!o??!r?aޕ0@)???!o??1r?aޕ0@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap!u;?ʃ??!?P~R?:@)????߆??1??u???/@:Preprocessing2F
Iterator::ModelϠ?????!??<*B<@)?tx㧁?1??g??'@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice???s????!?.*2х%@)???s????1?.*2х%@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?ɐ??!??pu??Q@)F?Swew?1R%?b?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor???B??r?!??uG%@)???B??r?1??uG%@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 33.5% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9???? 	@I3H_h?7X@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??S??????S????!??S????      ??!       "      ??!       *      ??!       2	7?C$??7?C$??!7?C$??:      ??!       B      ??!       J	??y??w????y??w??!??y??w??R      ??!       Z	??y??w????y??w??!??y??w??b      ??!       JCPU_ONLYY???? 	@b q3H_h?7X@