import { formatRelativeTime } from '../utils/formatters.js';

export default {
    template: `
        <div v-if="event" class="modal active fixed inset-0 bg-black bg-opacity-75 items-center justify-center z-50 p-4" @click.self="close">
            <div class="bg-gray-800 rounded-lg p-6 max-w-4xl w-full max-h-[90vh] overflow-y-auto">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h3 class="text-xl font-bold">
                            <span @click="filterBy('event_type', event.event_type)" class="hover:text-blue-400 hover:underline cursor-pointer">{{ event.event_type }}</span>
                            -
                            <span @click="filterBy('camera_id', event.camera_id)" class="hover:text-blue-400 hover:underline cursor-pointer">{{ event.camera_id }}</span>
                        </h3>
                        <p class="text-gray-400 text-sm">{{ new Date(event.timestamp).toLocaleString() }}</p>
                    </div>
                    <button @click="close" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-2 gap-8">
                    <div class="space-y-6">
                        <section>
                            <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Details</h4>
                            <div class="bg-gray-700/50 rounded-lg p-4 space-y-3">
                                <div class="flex justify-between">
                                    <span class="text-gray-400">Person</span>
                                    <span v-if="personDisplay !== 'Unidentified'" @click="filterBy('person_name', event.person_name)" class="font-medium text-blue-400 hover:underline cursor-pointer">{{ personDisplay }}</span>
                                    <span v-else class="font-medium text-blue-400">{{ personDisplay }}</span>
                                </div>
                                <div v-if="event.track_id" class="flex justify-between">
                                    <span class="text-gray-400">Track</span>
                                    <span @click="filterBy('track_id', event.track_id)" class="font-medium font-mono hover:text-blue-400 hover:underline cursor-pointer">{{ event.track_id }}</span>
                                </div>
                                <div class="flex justify-between">
                                    <span class="text-gray-400">Confidence</span>
                                    <span class="font-medium">{{ confidenceDisplay }}</span>
                                </div>
                            </div>
                        </section>

                        <!-- Best Match section -->
                        <section v-if="hasBestMatch">
                            <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Best Match ({{ bestMatchScore }}%)</h4>
                            <div class="bg-gray-700/50 rounded-lg p-4 text-center">
                                <img :src="'/api/v1/faces/image/' + event.extra_data.best_match_face_id"
                                     class="w-32 h-32 object-cover rounded shadow-lg mx-auto border border-gray-600"
                                     alt="Best Match Face">
                                <p class="mt-2 text-sm">{{ event.extra_data.best_match_person_name || 'Unknown' }}</p>
                            </div>
                        </section>

                        <!-- Reference Face section -->
                        <section v-if="hasReferenceFace">
                            <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Reference Face</h4>
                            <div class="bg-gray-700/50 rounded-lg p-4 text-center">
                                <img :src="'/api/v1/faces/image/' + event.extra_data.face_id"
                                     class="w-32 h-32 object-cover rounded shadow-lg mx-auto border border-gray-600"
                                     alt="Reference Face">
                            </div>
                        </section>

                        <!-- Gallery Match section -->
                        <section v-if="hasGalleryMatch">
                            <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Gallery Match</h4>
                            <div class="bg-gray-700/50 rounded-lg p-4 text-center">
                                <img :src="'/api/v1/reid-embeddings/' + event.extra_data.original_reid_embedding_id + '/image'"
                                     class="max-w-48 max-h-48 object-contain rounded shadow-lg mx-auto border border-gray-600"
                                     alt="Gallery Match Crop">
                            </div>
                        </section>

                        <section>
                            <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Metadata</h4>
                            <pre class="bg-gray-900 text-green-400 p-4 rounded-lg text-xs overflow-x-auto border border-gray-700 font-mono">{{ formattedExtra }}</pre>
                        </section>
                    </div>
                    <div class="space-y-4">
                        <h4 class="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2">Snapshot</h4>
                        <div v-if="event.snapshot_path" class="space-y-2">
                            <img :src="'/api/v1/events/' + event.id + '/snapshot'"
                                 class="max-w-full h-auto rounded-lg shadow-2xl mx-auto border border-gray-700"
                                 alt="Event Snapshot">
                        </div>
                        <p v-else class="text-sm text-gray-500 italic">No snapshot available</p>
                    </div>
                </div>

                <div class="mt-6 pt-4 border-t border-gray-700">
                    <button @click="close" class="w-full bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded">
                        Close
                    </button>
                </div>
            </div>
        </div>
    `,
    props: {
        event: {
            type: Object,
            default: null
        }
    },
    emits: ['close', 'filter-events'],
    computed: {
        personDisplay() {
            if (!this.event) return '';
            return this.event.person_name || (this.event.person_id ? `ID: ${this.event.person_id}` : 'Unidentified');
        },
        confidenceDisplay() {
            if (!this.event) return 'N/A';
            return (this.event.confidence !== null && this.event.confidence !== undefined)
                ? (this.event.confidence * 100).toFixed(1) + '%' : 'N/A';
        },
        formattedExtra() {
            if (!this.event) return '{}';
            return this.event.extra_data ? JSON.stringify(this.event.extra_data, null, 2) : '{}';
        },
        hasBestMatch() {
            return this.event && this.event.event_type === 'unidentified_face_saved'
                && this.event.extra_data && this.event.extra_data.best_match_face_id;
        },
        bestMatchScore() {
            if (!this.hasBestMatch) return 0;
            return (this.event.extra_data.best_match_score * 100).toFixed(0);
        },
        hasReferenceFace() {
            return this.event && this.event.event_type === 'face_match'
                && this.event.extra_data && this.event.extra_data.face_id;
        },
        hasGalleryMatch() {
            return this.event && this.event.event_type === 'reid_match'
                && this.event.extra_data && this.event.extra_data.original_reid_embedding_id;
        }
    },
    methods: {
        close() {
            this.$emit('close');
        },
        filterBy(field, value) {
            if (!value) return;
            this.$emit('filter-events', { field, value });
            this.close();
        }
    }
};
