import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ImageList } from './models/image-list';
import { InferenceRequest } from './models/inference-request';
import { InferenceResult } from './models/inference-result';

@Injectable({
  providedIn: 'root'
})
export class InferenceService {
  readonly baseUrl = "http://localhost:5000"
  constructor(private httpClient: HttpClient) { }
  getImagesNames() : Observable<ImageList>{
    return this.httpClient.get<ImageList>(this.baseUrl+"/images-names")
  }

  doInference(body: InferenceRequest): Observable<InferenceResult>{
    return this.httpClient.put<InferenceResult>(this.baseUrl+"/vqa-inference", body)
  }

}
