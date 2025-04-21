import {NgModule} from '@angular/core';
import {CommonModule} from '@angular/common';

import {SignInRoutingModule} from './sign-in-routing.module';
import {SignInComponent} from './Components/sign-in/sign-in.component';
import {AngularMaterialModule} from "../../../../Core/DesignModules/angular-material/angular-material.module";


@NgModule({
    declarations: [
        SignInComponent
    ],
    imports: [
        CommonModule,
        SignInRoutingModule,
        AngularMaterialModule
    ]
})
export class SignInModule {
}
