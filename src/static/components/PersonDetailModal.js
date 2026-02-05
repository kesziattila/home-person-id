export default {
    template: `
        <div v-if="visible" @click.self="close" class="modal active fixed inset-0 bg-black bg-opacity-75 items-center justify-center z-50 p-4">
            <div class="bg-gray-800 rounded-lg p-6 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
                <div class="flex justify-between items-start mb-4">
                    <div>
                        <h3 class="text-xl font-bold">{{ name }}</h3>
                        <p class="text-gray-400 text-sm">Created: {{ createdDate }}</p>
                    </div>
                    <button @click="close" class="text-gray-400 hover:text-white text-2xl">&times;</button>
                </div>

                <div v-if="loading" class="flex justify-center items-center p-16">
                    <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500"></div>
                </div>
                <div v-else-if="error" class="text-red-500">{{ error }}</div>
                <div v-else>
                    <!-- Face Images Gallery -->
                    <div class="mb-6">
                        <div class="flex justify-between items-center mb-3">
                            <h4 class="font-semibold">Face Images</h4>
                            <button @click="showAddImages = !showAddImages" class="bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded text-sm">
                                {{ showAddImages ? 'Cancel' : '+ Add Images' }}
                            </button>
                        </div>
                        <div class="flex flex-wrap gap-3">
                            <div v-for="img in images" :key="img.id" class="relative group">
                                <img v-if="img.has_image" :src="'/api/v1/faces/image/' + img.id" class="face-image" @click="$emit('preview-image', '/api/v1/faces/image/' + img.id)">
                                <div v-else class="face-placeholder text-xs">No file</div>
                                <div class="absolute -top-2 -right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                                    <button @click.stop="deleteImage(img.id)" class="bg-red-600 hover:bg-red-700 text-white rounded-full w-6 h-6 text-xs">&times;</button>
                                </div>
                            </div>
                            <p v-if="images.length === 0" class="text-gray-500 text-sm">No face images stored</p>
                        </div>
                    </div>

                    <!-- Add Images Section -->
                    <div v-if="showAddImages" class="mb-6">
                        <h4 class="font-semibold mb-3">Upload New Images</h4>
                        <div class="upload-zone mb-3" 
                             @click="triggerFileInput"
                             @dragover.prevent @dragleave.prevent @drop.prevent="handleDrop">
                            <input type="file" ref="fileInput" @change="handleFileSelect" multiple accept="image/*" class="hidden">
                            <p class="text-gray-400">Drop images here or click to select</p>
                        </div>
                        <div v-if="filesToUpload.length > 0" class="mb-3 text-sm text-gray-400">
                            Selected: {{ filesToUpload.map(f => f.name).join(', ') }}
                        </div>
                        <div class="flex gap-2">
                            <button @click="uploadImages" class="bg-green-600 hover:bg-green-700 px-4 py-2 rounded text-sm">Upload Images</button>
                        </div>
                        <div v-if="uploadStatus" class="mt-2 text-sm" :class="uploadStatus.error ? 'text-red-500' : 'text-green-500'">
                            {{ uploadStatus.message }}
                        </div>
                    </div>

                    <!-- Actions -->
                    <div class="border-t border-gray-700 pt-4 flex justify-between">
                        <button @click="$emit('show-timeline', personId)" class="text-blue-500 hover:text-blue-400 text-sm">View Timeline</button>
                        <button @click="deletePerson" class="text-red-500 hover:text-red-400 text-sm">Delete Person</button>
                    </div>
                </div>
            </div>
        </div>
    `,
    props: ['personId'],
    emits: ['close', 'preview-image', 'show-timeline', 'person-deleted'],
    data() {
        return {
            visible: false,
            loading: false,
            error: null,
            name: '',
            createdDate: '',
            images: [],
            showAddImages: false,
            filesToUpload: [],
            uploadStatus: null,
        };
    },
    watch: {
        async personId(newId) {
            if (newId) {
                this.visible = true;
                await this.loadPersonDetails(newId);
            } else {
                this.visible = false;
            }
        }
    },
    methods: {
        close() {
            this.$emit('close');
        },
        resetState() {
            this.name = '';
            this.createdDate = '';
            this.images = [];
            this.showAddImages = false;
            this.filesToUpload = [];
            this.uploadStatus = null;
            this.error = null;
        },
        async loadPersonDetails(personId) {
            this.loading = true;
            this.resetState();
            try {
                const response = await fetch(`/api/v1/persons/${personId}/detail`);
                if (!response.ok) throw new Error('Failed to load person details');
                const detail = await response.json();
                this.name = detail.name;
                this.createdDate = new Date(detail.created_at).toLocaleString();
                this.images = detail.images || [];
            } catch (e) {
                this.error = e.message;
                console.error(e);
            } finally {
                this.loading = false;
            }
        },
        async deleteImage(imageId) {
            if (!confirm('Are you sure you want to delete this face image?')) return;
            try {
                const response = await fetch(`/api/v1/persons/${this.personId}/images/${imageId}`, { method: 'DELETE' });
                if (response.ok) {
                    await this.loadPersonDetails(this.personId); // Refresh
                } else {
                    alert('Failed to delete image');
                }
            } catch (e) {
                alert('Error deleting image: ' + e.message);
            }
        },
        async deletePerson() {
            if (!confirm(`Are you sure you want to delete "${this.name}"? This action cannot be undone.`)) return;
            try {
                const response = await fetch(`/api/v1/persons/${this.personId}`, { method: 'DELETE' });
                if (response.ok) {
                    this.$emit('person-deleted');
                    this.close();
                } else {
                    alert('Failed to delete person');
                }
            } catch (e) {
                alert('Error deleting person: ' + e.message);
            }
        },
        triggerFileInput() {
            this.$refs.fileInput.click();
        },
        handleDrop(event) {
            this.filesToUpload = Array.from(event.dataTransfer.files);
        },
        handleFileSelect(event) {
            this.filesToUpload = Array.from(event.target.files);
        },
        async uploadImages() {
            if (this.filesToUpload.length === 0) {
                this.uploadStatus = { message: 'Please select files to upload.', error: true };
                return;
            }
            this.uploadStatus = { message: 'Uploading...', error: false };
            
            const formData = new FormData();
            for (const file of this.filesToUpload) {
                formData.append('files', file);
            }

            try {
                const response = await fetch(`/api/v1/persons/${this.personId}/images`, {
                    method: 'POST',
                    body: formData
                });
                const result = await response.json();
                if (response.ok) {
                    this.uploadStatus = { message: result.message, error: false };
                    this.filesToUpload = [];
                    await this.loadPersonDetails(this.personId); // Refresh
                } else {
                    this.uploadStatus = { message: result.detail || 'Upload failed', error: true };
                }
            } catch (e) {
                this.uploadStatus = { message: 'Upload failed: ' + e.message, error: true };
            }
        }
    }
};
