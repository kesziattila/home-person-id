export default {
    template: `
        <div v-if="visible" class="modal active fixed inset-0 bg-black bg-opacity-75 items-center justify-center z-50 p-4">
            <div class="bg-gray-800 rounded-lg p-6 max-w-md w-full">
                <h3 class="text-xl font-bold mb-4">Assign Face to Person</h3>
                <div v-if="loading" class="flex justify-center items-center p-8">
                    <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-500"></div>
                </div>
                <div v-else-if="error" class="text-red-500">{{ error }}</div>
                <div v-else>
                    <div class="mb-4">
                        <img :src="'/api/v1/unidentified-faces/' + faceId + '/image'" class="w-32 h-32 object-cover rounded mx-auto mb-2">
                        <div class="text-center mb-2" v-html="qualityInfo"></div>
                        <div class="text-sm text-gray-400 text-center mb-4">{{ faceInfo }}</div>
                    </div>
                    <div class="mb-4">
                        <label class="block text-sm font-medium mb-2">Select Person</label>
                        <select v-model="selectedPersonId" class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2">
                            <option value="">-- Select a person --</option>
                            <option v-for="person in persons" :key="person.id" :value="person.id">
                                {{ person.name }}
                            </option>
                        </select>
                    </div>
                    <div class="flex gap-2">
                        <button @click="confirmAssignment" class="flex-1 bg-green-600 hover:bg-green-700 px-4 py-2 rounded">
                            Assign
                        </button>
                        <button @click="close" class="flex-1 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded">
                            Cancel
                        </button>
                    </div>
                    <div v-if="status" class="mt-4 text-sm" :class="status.error ? 'text-red-500' : 'text-green-500'">
                        {{ status.message }}
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['faceId'],
    emits: ['close', 'face-assigned'],
    data() {
        return {
            visible: false,
            loading: false,
            error: null,
            faceDetails: null,
            persons: [],
            selectedPersonId: '',
            status: null,
        };
    },
    computed: {
        qualityInfo() {
            if (!this.faceDetails) return '';
            const { quality_score, blur_score } = this.faceDetails;
            const isGood = this.isGoodForLearning(quality_score, blur_score);
            const qualityText = `Q:${(quality_score*100).toFixed(0)}% S:${(blur_score*100).toFixed(0)}%`;
            if (isGood) {
                return `<span class="inline-block bg-green-600 text-xs px-3 py-1 rounded">✓ Good for learning</span>
                        <span class="text-xs text-gray-500 ml-2">${qualityText}</span>`;
            }
            return `<span class="inline-block bg-yellow-600 text-xs px-3 py-1 rounded">⚠ Low quality</span>
                    <span class="text-xs text-gray-500 ml-2">${qualityText}</span>`;
        },
        faceInfo() {
            if (!this.faceDetails) return '';
            let info = `Camera: ${this.faceDetails.camera_id}`;
            if (this.faceDetails.best_match_person_name) {
                info += ` | Best match: ${this.faceDetails.best_match_person_name} (${(this.faceDetails.best_match_score * 100).toFixed(0)}%)`;
            }
            return info;
        }
    },
    watch: {
        async faceId(newId) {
            if (newId) {
                this.visible = true;
                await this.loadData(newId);
            }
        }
    },
    methods: {
        close() {
            this.visible = false;
            this.resetState();
            this.$emit('close');
        },
        resetState() {
            this.faceDetails = null;
            this.persons = [];
            this.selectedPersonId = '';
            this.status = null;
            this.error = null;
        },
        async loadData(faceId) {
            this.loading = true;
            this.error = null;
            try {
                const [faceRes, personsRes] = await Promise.all([
                    fetch(`/api/v1/unidentified-faces/${faceId}`),
                    fetch('/api/v1/persons')
                ]);
                if (!faceRes.ok || !personsRes.ok) throw new Error('Failed to load data.');
                
                this.faceDetails = await faceRes.json();
                this.persons = await personsRes.json();
                
                // Pre-select best match if available
                if (this.faceDetails.best_match_person_id) {
                    this.selectedPersonId = this.faceDetails.best_match_person_id;
                }
            } catch (e) {
                this.error = e.message;
                console.error(e);
            } finally {
                this.loading = false;
            }
        },
        async confirmAssignment() {
            if (!this.selectedPersonId) {
                this.status = { message: 'Please select a person.', error: true };
                return;
            }
            this.status = { message: 'Assigning...', error: false };
            try {
                const response = await fetch(`/api/v1/unidentified-faces/${this.faceId}/assign`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ person_id: parseInt(this.selectedPersonId) })
                });
                const result = await response.json();
                if (response.ok) {
                    this.status = { message: result.message, error: false };
                    setTimeout(() => {
                        this.$emit('face-assigned');
                        this.close();
                    }, 1000);
                } else {
                    this.status = { message: result.detail || 'Failed to assign face.', error: true };
                }
            } catch (e) {
                this.status = { message: 'Error: ' + e.message, error: true };
            }
        },
        isGoodForLearning(quality, sharpness) {
            const LEARNING_QUALITY_THRESHOLD = 0.6;
            const LEARNING_SHARPNESS_THRESHOLD = 0.5;
            return sharpness >= LEARNING_SHARPNESS_THRESHOLD && quality >= LEARNING_QUALITY_THRESHOLD;
        }
    }
};
