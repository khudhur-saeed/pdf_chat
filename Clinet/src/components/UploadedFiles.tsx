import { useState, useEffect } from "react";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { FileText, Trash2, RefreshCw, FolderOpen, Check } from "lucide-react";

const API_BASE_URL = "http://localhost:8000";

interface UploadedFile {
  filename: string;
  size: number;
  uploaded_at: number;
}

interface UploadedFilesProps {
  onFileDeleted?: () => void;
  onFileSelected?: (filename: string) => void;
  selectedFile?: string;
  refreshTrigger?: number;
}

export function UploadedFiles({ onFileDeleted, onFileSelected, selectedFile, refreshTrigger }: UploadedFilesProps) {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [loading, setLoading] = useState(false);
  const [deleting, setDeleting] = useState<string | null>(null);
  const [selecting, setSelecting] = useState<string | null>(null);

  const fetchFiles = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${API_BASE_URL}/uploaded-files`);
      const data = await response.json();
      setFiles(data.files || []);
    } catch (error) {
      console.error("Failed to fetch files:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, [refreshTrigger]);

  const handleSelectFile = async (filename: string) => {
    setSelecting(filename);
    try {
      const response = await fetch(`${API_BASE_URL}/select-pdf/${encodeURIComponent(filename)}`, {
        method: "POST",
      });
      
      if (response.ok) {
        const data = await response.json();
        onFileSelected?.(filename);
      } else {
        const error = await response.json();
        alert(error.detail || "Failed to select PDF");
      }
    } catch (error) {
      console.error("Failed to select file:", error);
      alert("Failed to select PDF");
    } finally {
      setSelecting(null);
    }
  };

  const handleDelete = async (filename: string) => {
    if (!confirm(`Are you sure you want to delete "${filename}"?`)) {
      return;
    }

    setDeleting(filename);
    try {
      const response = await fetch(`${API_BASE_URL}/uploaded-files/${encodeURIComponent(filename)}`, {
        method: "DELETE",
      });
      
      if (response.ok) {
        setFiles(files.filter(f => f.filename !== filename));
        onFileDeleted?.();
      } else {
        const error = await response.json();
        alert(error.detail || "Failed to delete file");
      }
    } catch (error) {
      console.error("Failed to delete file:", error);
      alert("Failed to delete file");
    } finally {
      setDeleting(null);
    }
  };

  const formatFileSize = (bytes: number) => {
    if (bytes < 1024) return bytes + " B";
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
    return (bytes / (1024 * 1024)).toFixed(1) + " MB";
  };

  const formatDate = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleDateString() + " " + date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between px-2 py-1">
        <div className="flex items-center gap-2 text-xs font-medium text-sidebar-foreground/70 uppercase tracking-wider">
          <FolderOpen className="h-3 w-3" />
          Uploaded PDFs
        </div>
        <Button
          variant="ghost"
          size="sm"
          onClick={fetchFiles}
          disabled={loading}
          className="h-6 w-6 p-0 hover:bg-sidebar-accent"
          title="Refresh"
        >
          <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
        </Button>
      </div>

      {files.length === 0 ? (
        <div className="px-2 py-3 text-xs text-sidebar-foreground/50 text-center">
          No PDFs uploaded yet
        </div>
      ) : (
        <ScrollArea className="max-h-[200px]">
          <div className="space-y-1">
            {files.map((file) => (
              <div
                key={file.filename}
                className={`group flex items-start gap-2 px-2 py-2 rounded-md cursor-pointer transition-colors ${
                  selectedFile === file.filename
                    ? "bg-orange-500/20 border border-orange-500"
                    : "hover:bg-sidebar-accent/50"
                }`}
              >
                <FileText className="h-4 w-4 text-orange-500 flex-shrink-0 mt-0.5" />
                <div 
                  className="flex-1 min-w-0"
                  onClick={() => handleSelectFile(file.filename)}
                >
                  <p className="text-xs font-medium text-sidebar-foreground truncate" title={file.filename}>
                    {file.filename}
                  </p>
                  <p className="text-xs text-sidebar-foreground/50">
                    {formatFileSize(file.size)}
                  </p>
                </div>
                <div className="flex items-center gap-1">
                  {selectedFile === file.filename && (
                    <Check className="h-4 w-4 text-green-500 flex-shrink-0" />
                  )}
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleDelete(file.filename)}
                    disabled={deleting === file.filename}
                    className="h-6 w-6 p-0 opacity-0 group-hover:opacity-100 hover:bg-red-500/20 hover:text-red-500 transition-opacity"
                    title="Delete file"
                  >
                    {deleting === file.filename ? (
                      <RefreshCw className="h-3 w-3 animate-spin" />
                    ) : (
                      <Trash2 className="h-3 w-3" />
                    )}
                  </Button>
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>
      )}
    </div>
  );
}
