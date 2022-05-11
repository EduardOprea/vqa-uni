import { Component, OnInit } from '@angular/core';
import { InferenceService } from './inference.service';
import { ImageList } from './models/image-list';
import { InferenceRequest } from './models/inference-request';
import { InferenceResult } from './models/inference-result';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit {
  title = 'vqa-demo';
  imgRootPath = "/assets/test-images/"
  imgName = "2.jpg"
  imgPath = this.imgRootPath + this.imgName;
  question: string | undefined
  imageList: ImageList | undefined
  inferenceResult: InferenceResult | undefined
  waitingResult: boolean = false
  

  constructor(private inferenceService: InferenceService){

  }
  ngOnInit(): void {
    this.inferenceService.getImagesNames().subscribe((data:ImageList) => {
      console.log("succes get image names")  
      console.log(data)
      this.imageList = data
    }, err => {
      console.log("error get image names")
    })
  }
  

  onNextPhoto():void{
    let randomIndex = Math.floor(Math.random() * (this.imageList!.imagesFilenames.length));
    this.imgName = this.imageList!.imagesFilenames[randomIndex]
    this.imgPath = this.imgRootPath + this.imgName
  }

  onSendQuestionImagePair():void{
    this.waitingResult = true 
    let body : InferenceRequest = {
      imageName: this.imgName,
      question: this.question!
    }
    this.inferenceResult = undefined
    console.log("start doing inference")
    this.inferenceService.doInference(body).subscribe((result: InferenceResult) => {
      console.log("succes vqa inference")
      console.log(result)
      this.inferenceResult = result
      this.waitingResult = false
    }, err => {
      console.log("error")
      this.inferenceResult = undefined
      this.waitingResult = false
    })
  }
}
