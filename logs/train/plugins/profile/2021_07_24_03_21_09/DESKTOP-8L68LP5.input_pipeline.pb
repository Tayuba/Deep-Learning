	</?z??</?z??!</?z??	M?sr?)3@M?sr?)3@!M?sr?)3@"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:</?z??¥c?3???A??0?qf??Y????w???rEagerKernelExecute 0*	G????.v@2F
Iterator::ModeliQ?????!ScJ?ݓU@)?%Z?x??1>"???TT@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat]???Ա??!S+a@){Ic?????1???g+?@:Preprocessing2U
Iterator::Model::ParallelMapV2X???!??!HT???@)X???!??1HT???@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap^=?1X??!1?-m?@)?)H??1:׾?:R@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlicepw?n??|?!O?8????)pw?n??|?1O?8????:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip????????!h??a+@)???G?v?12y???(??:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??zp?!??xm????)??zp?1??xm????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 19.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*moderate2s5.7 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9M?sr?)3@I?c#?5T@Zno>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	¥c?3???¥c?3???!¥c?3???      ??!       "      ??!       *      ??!       2	??0?qf????0?qf??!??0?qf??:      ??!       B      ??!       J	????w???????w???!????w???R      ??!       Z	????w???????w???!????w???b      ??!       JCPU_ONLYYM?sr?)3@b q?c#?5T@