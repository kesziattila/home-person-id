import { formatRelativeTime } from '../utils/formatters.js';

export default {
    template: `
        <div>
            <div class="mb-4 flex justify-between items-center">
                <div>
                    <h2 class="text-2xl font-bold mb-2">Unidentified Faces</h2>
                    <p class="text-gray-400 text-sm">Faces detected below the recognition threshold for manual review</p>
                </div>
                <div class="flex flex-wrap gap-2 items-center">
                    <select v-model="cameraFilter" @change="loadFaces" class="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                        <option value="">All Cameras</option>
                        <option v-for="(count, camera) in cameraSummary" :key="camera" :value="camera">
                            {{ camera }} ({{ count }})
                        </option>
                    </select>
                    <select v-model="qualityFilter" @change="loadFaces" class="bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm">
                        <option value="0">All Quality</option>
                        <option value="0.4">40%+</option>
                        <option value="0.5">50%+</option>
                        <option value="0.6">60%+</option>
                    </select>
                    <label class="flex items-center gap-2 text-sm cursor-pointer">
                        <input type="checkbox" v-model="learningOnly" @change="loadFaces" class="rounded">
                        <span>Learning only</span>
                    </label>
                    <button @click="loadFaces" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm">
                        Refresh
                    </button>
                </div>
            </div>
            <div class="mb-4 text-sm text-gray-400">{{ summaryText }}</div>
            <div v-if="loading" class="flex justify-center items-center p-16">
                <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
            </div>
            <div v-else-if="error" class="text-red-500 text-center py-20">
                {{ error }}
            </div>
            <div v-else-if="filteredFaces.length === 0" class="text-center py-20 text-gray-500">
                <p>No unidentified faces to review</p>
                <p class="text-sm mt-2">Faces that fall below the recognition threshold will appear here</p>
            </div>
            <div v-else class="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
                <div v-for="face in filteredFaces" :key="face.id" class="bg-gray-800 rounded-lg overflow-hidden border border-gray-700 hover:border-blue-500 transition-colors">
                    <div class="aspect-square bg-gray-900 relative">
                        <img :src="'/api/v1/unidentified-faces/' + face.id + '/image'"
                             class="w-full h-full object-cover cursor-pointer"
                             @click="previewFace(face.id)"
                             onerror="this.src='https://via.placeholder.com/200?text=No+Image'">
                        
                        <div v-if="isGoodForLearning(face.quality_score, face.blur_score)" class="absolute top-2 left-2 bg-green-600 text-xs px-2 py-1 rounded flex items-center gap-1">
                            <span>✓</span> Learning
                        </div>
                        <div v-else class="absolute top-2 left-2 bg-gray-600 text-xs px-2 py-1 rounded opacity-75">
                            Review
                        </div>

                        <div :class="getQualityColor(face.quality_score)" class="absolute top-2 right-2 text-xs px-2 py-1 rounded">
                            Q: {{ (face.quality_score * 100).toFixed(0) }}%
                        </div>
                    </div>
                    <div class="p-3">
                        <div class="text-xs text-gray-400 mb-1">{{ face.camera_id }}</div>
                        <div class="text-xs text-gray-500 mb-2">{{ formatRelativeTime(face.created_at) }}</div>
                        <div class="text-sm mb-3" v-html="getMatchInfo(face)"></div>
                        <div class="flex gap-2">
                            <button v-if="face.best_match_person_id"
                                    @click="quickAssign(face.id, face.best_match_person_id)"
                                    class="flex-1 bg-green-600 hover:bg-green-700 px-2 py-1 rounded text-xs"
                                    :title="'Confirm as ' + face.best_match_person_name">
                                Confirm
                            </button>
                            <button @click="showAssignModal(face.id)"
                                    class="flex-1 bg-blue-600 hover:bg-blue-700 px-2 py-1 rounded text-xs">
                                Assign
                            </button>
                            <button @click="dismissFace(face.id)"
                                    class="flex-1 bg-gray-600 hover:bg-gray-500 px-2 py-1 rounded text-xs">
                                Dismiss
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['autoRefresh'],
    emits: ['show-assign-face', 'face-assigned', 'preview-image'],
    data() {
        return {
            faces: [],
            cameraSummary: {},
            loading: true,
            error: null,
            cameraFilter: '',
            qualityFilter: '0',
            learningOnly: false,
            refreshInterval: null,
        };
    },
    computed: {
        filteredFaces() {
            let faces = this.faces;
            if (this.qualityFilter > 0) {
                faces = faces.filter(f => (f.quality_score || 0) >= this.qualityFilter);
            }
            if (this.learningOnly) {
                faces = faces.filter(f => this.isGoodForLearning(f.quality_score || 0, f.blur_score || 0));
            }
            return faces;
        },
        summaryText() {
            const total = Object.values(this.cameraSummary).reduce((a, b) => a + b, 0);
            const cameraCount = Object.keys(this.cameraSummary).length;
            let text = `Total: ${total} faces across ${cameraCount} camera(s)`;
            if (this.qualityFilter > 0 || this.learningOnly) {
                text += ` (showing ${this.filteredFaces.length} filtered)`;
            }
            return text;
        }
    },
    async created() {
        await this.loadFaces();
        if (this.autoRefresh) {
            this.refreshInterval = setInterval(this.loadFaces, 10000);
        }
    },
    beforeUnmount() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
    },
    watch: {
        autoRefresh(newVal) {
            if (newVal && !this.refreshInterval) {
                this.refreshInterval = setInterval(this.loadFaces, 10000);
            } else if (!newVal && this.refreshInterval) {
                clearInterval(this.refreshInterval);
                this.refreshInterval = null;
            }
        }
    },
    methods: {
        formatRelativeTime,
        async loadFaces() {
            this.loading = true;
            this.error = null;
            try {
                const [summaryRes, facesRes] = await Promise.all([
                    fetch('/api/v1/unidentified-faces/summary'),
                    fetch(`/api/v1/unidentified-faces?dismissed=false&limit=200&camera_id=${this.cameraFilter}`)
                ]);
                if (!summaryRes.ok || !facesRes.ok) throw new Error('Failed to fetch data');
                
                const summary = await summaryRes.json();
                this.cameraSummary = summary.by_camera || {};
                this.faces = await facesRes.json();

            } catch (e) {
                this.error = 'Error loading unidentified faces.';
                console.error(e);
            } finally {
                this.loading = false;
            }
        },
        isGoodForLearning(quality, sharpness) {
            const LEARNING_QUALITY_THRESHOLD = 0.6;
            const LEARNING_SHARPNESS_THRESHOLD = 0.5;
            return sharpness >= LEARNING_SHARPNESS_THRESHOLD && quality >= LEARNING_QUALITY_THRESHOLD;
        },
        getQualityColor(score) {
            if (score > 0.6) return 'bg-green-600';
            if (score > 0.4) return 'bg-yellow-600';
            return 'bg-red-600';
        },
        getMatchInfo(face) {
            if (face.best_match_person_name && face.best_match_score) {
                const scorePercent = (face.best_match_score * 100).toFixed(0);
                const scoreColor = face.best_match_score > 0.5 ? 'text-yellow-400' : 'text-gray-400';
                return `<span class="${scoreColor}">${face.best_match_person_name} (${scorePercent}%)</span>`;
            }
            return '<span class="text-gray-500">No match</span>';
        },
        previewFace(faceId) {
            this.$emit('preview-image', `/api/v1/unidentified-faces/${faceId}/image`);
        },
        async quickAssign(faceId, personId) {
            if (!confirm('Confirm this face assignment?')) return;
            try {
                const response = await fetch(`/api/v1/unidentified-faces/${faceId}/assign`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ person_id: personId })
                });
                if (response.ok) {
                    this.$emit('face-assigned');
                } else {
                    const result = await response.json();
                    alert(result.detail || 'Failed to assign face');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        },
        showAssignModal(faceId) {
            this.$emit('show-assign-face', faceId);
        },
        async dismissFace(faceId) {
            try {
                const response = await fetch(`/api/v1/unidentified-faces/${faceId}/dismiss`, {
                    method: 'POST'
                });
                if (response.ok) {
                    this.loadFaces(); // Refresh the list
                } else {
                    const result = await response.json();
                    alert(result.detail || 'Failed to dismiss face');
                }
            } catch (error) {
                alert('Error: ' + error.message);
            }
        }
    }
};
