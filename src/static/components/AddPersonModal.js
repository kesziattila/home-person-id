export default {
    template: `
        <div v-if="visible" @click.self="close" class="modal active fixed inset-0 bg-black bg-opacity-75 items-center justify-center z-50 p-4">
            <div class="bg-gray-800 rounded-lg p-6 max-w-md w-full">
                <h3 class="text-xl font-bold mb-4">Add New Person</h3>
                <form @submit.prevent="createPerson">
                    <div class="mb-4">
                        <label class="block text-sm font-medium mb-2">Name *</label>
                        <input type="text" v-model="name" required
                               class="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 focus:outline-none focus:border-blue-500"
                               placeholder="Enter person's name">
                    </div>
                    <div class="mb-4">
                        <label class="block text-sm font-medium mb-2">Face Images (optional)</label>
                        <div class="upload-zone" 
                             @click="triggerFileInput"
                             @dragover.prevent @dragleave.prevent @drop.prevent="handleDrop">
                            <input type="file" ref="fileInput" @change="handleFileSelect" multiple accept="image/*" class="hidden">
                            <p class="text-gray-400">Drop images here or click to select</p>
                        </div>
                        <div v-if="files.length > 0" class="mt-2 text-sm text-gray-400">
                            Selected: {{ files.map(f => f.name).join(', ') }}
                        </div>
                    </div>
                    <div class="flex gap-2">
                        <button type="submit" class="flex-1 bg-green-600 hover:bg-green-700 px-4 py-2 rounded">
                            Create Person
                        </button>
                        <button type="button" @click="close" class="flex-1 bg-gray-700 hover:bg-gray-600 px-4 py-2 rounded">
                            Cancel
                        </button>
                    </div>
                </form>
                <div v-if="status" class="mt-4 text-sm" :class="status.error ? 'text-red-500' : 'text-green-500'">
                    {{ status.message }}
                </div>
            </div>
        </div>
    `,
    props: {
        visible: {
            type: Boolean,
            default: false,
        }
    },
    emits: ['close', 'person-added'],
    data() {
        return {
            name: '',
            files: [],
            status: null,
        };
    },
    methods: {
        close() {
            this.resetState();
            this.$emit('close');
        },
        resetState() {
            this.name = '';
            this.files = [];
            this.status = null;
        },
        triggerFileInput() {
            this.$refs.fileInput.click();
        },
        handleDrop(event) {
            this.files = Array.from(event.dataTransfer.files);
        },
        handleFileSelect(event) {
            this.files = Array.from(event.target.files);
        },
        async createPerson() {
            if (!this.name.trim()) {
                this.status = { message: 'Please enter a name.', error: true };
                return;
            }
            this.status = { message: 'Creating person...', error: false };

            try {
                const createResp = await fetch('/api/v1/persons', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: this.name })
                });

                if (!createResp.ok) {
                    const error = await createResp.json();
                    throw new Error(error.detail || 'Failed to create person');
                }
                const person = await createResp.json();

                if (this.files.length > 0) {
                    this.status = { message: 'Uploading images...', error: false };
                    const formData = new FormData();
                    for (const file of this.files) {
                        formData.append('files', file);
                    }
                    const uploadResp = await fetch(`/api/v1/persons/${person.id}/images`, {
                        method: 'POST',
                        body: formData
                    });
                    const result = await uploadResp.json();
                    this.status = { message: `Person created! ${result.message}`, error: !uploadResp.ok };
                } else {
                    this.status = { message: 'Person created successfully!', error: false };
                }

                setTimeout(() => {
                    this.$emit('person-added');
                    this.close();
                }, 1500);

            } catch (e) {
                this.status = { message: e.message, error: true };
            }
        }
    }
};
